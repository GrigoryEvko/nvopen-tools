// Function: sub_2308710
// Address: 0x2308710
//
_QWORD *__fastcall sub_2308710(_QWORD *a1, __int64 a2, __int64 a3)
{
  _BYTE *v4; // rax
  _BYTE *v5; // rsi
  unsigned __int64 v6; // r13
  __int64 v7; // rax
  char *v8; // rcx
  size_t v9; // rbx
  char *v10; // r13

  v4 = *(_BYTE **)(a2 + 8);
  v5 = *(_BYTE **)a2;
  v6 = v4 - v5;
  if ( v4 == v5 )
  {
    v9 = 0;
    v8 = 0;
  }
  else
  {
    if ( v6 > 0x7FFFFFFFFFFFFFFCLL )
      sub_4261EA(a1, v5, a3);
    v7 = sub_22077B0(v6);
    v5 = *(_BYTE **)a2;
    v8 = (char *)v7;
    v4 = *(_BYTE **)(a2 + 8);
    v9 = (size_t)&v4[-*(_QWORD *)a2];
  }
  v10 = &v8[v6];
  if ( v4 != v5 )
    v8 = (char *)memmove(v8, v5, v9);
  *a1 = v8;
  a1[2] = v10;
  a1[1] = &v8[v9];
  return a1;
}
