// Function: sub_2BE3780
// Address: 0x2be3780
//
char *__fastcall sub_2BE3780(__int64 a1, char *a2, char *a3)
{
  char *v3; // r8
  char *v5; // rdx
  char *v6; // rax
  char *v7; // rax

  v3 = a2;
  if ( a3 == a2 )
    return a2;
  v5 = *(char **)(a1 + 8);
  if ( a3 != v5 )
  {
    v6 = (char *)memmove(a2, a3, v5 - a3);
    v5 = *(char **)(a1 + 8);
    v3 = v6;
  }
  v7 = &v3[v5 - a3];
  if ( v7 != v5 )
    *(_QWORD *)(a1 + 8) = v7;
  return v3;
}
