// Function: sub_1DD9130
// Address: 0x1dd9130
//
void *__fastcall sub_1DD9130(__int64 a1, __int64 *a2, char a3)
{
  char *v5; // rax
  char *v6; // rdx
  char *v7; // rsi
  unsigned int *v8; // rsi

  if ( *(_QWORD *)(a1 + 112) != *(_QWORD *)(a1 + 120) )
  {
    v5 = (char *)sub_1DD7680(a1, (__int64)a2);
    v6 = *(char **)(a1 + 120);
    v7 = v5 + 4;
    if ( v6 != v5 + 4 )
    {
      memmove(v5, v7, v6 - v7);
      v7 = *(char **)(a1 + 120);
    }
    v8 = (unsigned int *)(v7 - 4);
    *(_QWORD *)(a1 + 120) = v8;
    if ( a3 )
      sub_1D96570(*(unsigned int **)(a1 + 112), v8);
  }
  sub_1DD9100(*a2, a1);
  return sub_1DD90C0(a1 + 88, a2);
}
