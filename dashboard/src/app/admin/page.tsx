"use client";

import { useEffect, useState } from "react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Button } from "@/components/ui/button";
import { BrainCircuit, Search, User, Clock, Trash } from "lucide-react";

export default function AdminDashboardPage() {
  const [persons, setPersons] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState("ALL");

  const fetchPersons = () => {
    fetch("http://localhost:8000/api/persons")
      .then((res) => res.json())
      .then((data) => {
        if (data.status === "success") {
          setPersons(data.data);
        }
      })
      .catch((err) => console.error("Failed to fetch persons:", err))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    fetchPersons();
  }, []);

  const updateCategory = async (personId: string, newCategory: string) => {
    try {
      const res = await fetch(`http://localhost:8000/api/persons/${personId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ category: newCategory })
      });
      const data = await res.json();
      if (data.status === 'success') {
        fetchPersons();
      } else {
        alert("Failed to update category: " + data.message);
      }
    } catch (err) {
      console.error(err);
    }
  };

  const deletePerson = async (personId: string) => {
    if (!confirm("Are you sure you want to remove this subject from the database?")) return;
    try {
      const res = await fetch(`http://localhost:8000/api/persons/${personId}`, {
        method: 'DELETE'
      });
      const data = await res.json();
      if (data.status === 'success') {
        fetchPersons();
      } else {
        alert("Failed to delete person: " + data.message);
      }
    } catch (err) {
      console.error(err);
    }
  };

  const filteredPersons = filter === "ALL" 
    ? persons 
    : persons.filter(p => p.category.includes(filter) || (filter === 'VIP' && p.category.includes('VIP')));

  return (
    <div className="flex h-full flex-col p-6 space-y-6 overflow-y-auto">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Admin Dashboard</h1>
          <p className="text-muted-foreground mt-1">System overview and high-level analytics.</p>
        </div>
      </div>
      {/* Enrolled Persons Directory */}
      <Card className="border-border/50 bg-secondary/20 flex-1">
        <div className="p-6 border-b border-border/50 flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
          <div>
            <h3 className="font-semibold text-lg flex items-center gap-2">
              <User className="w-5 h-5 text-primary" />
              Biometric Identity Database
            </h3>
            <p className="text-sm text-muted-foreground mt-1">Manage and view all enrolled subjects.</p>
          </div>
          <div className="flex gap-2 flex-wrap">
            {["ALL", "VIP", "STAFF", "WANTED", "SUSPECT"].map((cat) => (
              <Button 
                key={cat}
                variant={filter === cat ? "default" : "outline"}
                size="sm"
                onClick={() => setFilter(cat)}
                className={`text-xs ${filter === cat ? "bg-primary text-primary-foreground shadow-[0_0_10px_rgba(var(--primary),0.3)]" : "text-muted-foreground border-border/50 hover:text-foreground"}`}
              >
                {cat}
              </Button>
            ))}
          </div>
        </div>
        <div className="p-0">
          <Table>
            <TableHeader className="bg-secondary/50">
              <TableRow className="border-border/50 hover:bg-transparent">
                <TableHead className="w-[120px]">Subject ID</TableHead>
                <TableHead>Full Name</TableHead>
                <TableHead className="w-[150px]">Category</TableHead>
                <TableHead>Observations</TableHead>
                <TableHead>Enrolled At</TableHead>
                <TableHead className="text-right">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {loading ? (
                <TableRow>
                  <TableCell colSpan={6} className="text-center py-8 text-muted-foreground">Loading database...</TableCell>
                </TableRow>
              ) : filteredPersons.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={6} className="text-center py-8 text-muted-foreground">No subjects found in this category.</TableCell>
                </TableRow>
              ) : (
                filteredPersons.map((person) => {
                  const date = new Date(person.enrolled_at * 1000).toLocaleString();
                  const isThreat = person.category === "WANTED" || person.category === "SUSPECT";
                  
                  return (
                    <TableRow key={person.person_id} className="border-border/50 hover:bg-secondary/40 transition-colors">
                      <TableCell className="font-mono text-xs text-muted-foreground">{person.person_id}</TableCell>
                      <TableCell className="font-medium text-foreground">{person.name}</TableCell>
                      <TableCell>
                        <select 
                          value={person.category}
                          onChange={(e) => updateCategory(person.person_id, e.target.value)}
                          className={`bg-transparent border border-border/50 rounded text-xs px-2 py-1 font-semibold outline-none cursor-pointer hover:border-primary/50 transition-colors
                            ${isThreat ? 'text-destructive' : ''}
                            ${person.category.includes('VIP') ? 'text-primary' : ''}
                            ${person.category === 'STAFF' ? 'text-green-500' : ''}
                          `}
                        >
                          <option value="VIP">VIP</option>
                          <option value="STAFF">STAFF</option>
                          <option value="SUSPECT">SUSPECT</option>
                          <option value="WANTED">WANTED</option>
                          <option value="UNKNOWN">UNKNOWN</option>
                        </select>
                      </TableCell>
                      <TableCell>
                        <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
                          <BrainCircuit className="w-3.5 h-3.5" />
                          {person.face_count} enc.
                        </span>
                      </TableCell>
                      <TableCell className="text-xs text-muted-foreground">
                        <span className="flex items-center gap-1.5"><Clock className="w-3.5 h-3.5" />{date}</span>
                      </TableCell>
                      <TableCell className="text-right">
                        <Button variant="ghost" size="icon" onClick={() => deletePerson(person.person_id)} className="h-8 w-8 text-muted-foreground hover:text-destructive hover:bg-destructive/10">
                          <Trash className="w-4 h-4" />
                        </Button>
                      </TableCell>
                    </TableRow>
                  );
                })
              )}
            </TableBody>
          </Table>
        </div>
      </Card>

    </div>
  );
}
