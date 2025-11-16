// Function: sub_A73B90
// Address: 0xa73b90
//
char __fastcall sub_A73B90(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 *v3; // r13
  size_t v4; // rdx
  __int64 *v5; // r14
  __int64 v6; // rax
  __int64 *v7; // rbx
  int v8; // esi
  __int64 *v9; // r15
  __int64 *i; // rbx
  int v11; // esi
  __int64 *v13; // [rsp+8h] [rbp-38h]

  v3 = a2;
  *(_DWORD *)(a1 + 8) = a3;
  v4 = 8 * a3;
  *(_QWORD *)a1 = 0;
  v13 = &a2[v4 / 8];
  *(_OWORD *)(a1 + 12) = 0;
  *(_OWORD *)(a1 + 28) = 0;
  if ( v4 )
  {
    memmove((void *)(a1 + 48), a2, v4);
    v5 = (__int64 *)sub_A73280(a2);
    v6 = sub_A73290(a2);
    v7 = (__int64 *)v6;
    if ( v5 != (__int64 *)v6 )
      goto LABEL_5;
  }
  else
  {
    v5 = (__int64 *)sub_A73280(v13);
    v6 = sub_A73290(v13);
    v7 = (__int64 *)v6;
    if ( v5 == (__int64 *)v6 )
      return v6;
    do
    {
LABEL_5:
      while ( 1 )
      {
        LOBYTE(v6) = sub_A71840((__int64)v5);
        if ( !(_BYTE)v6 )
          break;
        if ( v7 == ++v5 )
          goto LABEL_9;
      }
      LODWORD(v6) = sub_A71AE0(v5);
      v8 = v6 + 7;
      if ( (int)v6 >= 0 )
        v8 = v6;
      ++v5;
      LODWORD(v6) = 1 << ((int)v6 % 8);
      *(_BYTE *)(a1 + (v8 >> 3) + 12) |= v6;
    }
    while ( v7 != v5 );
  }
LABEL_9:
  while ( v13 != v3 )
  {
    v9 = (__int64 *)sub_A73280(v3);
    v6 = sub_A73290(v3);
    for ( i = (__int64 *)v6; i != v9; *(_BYTE *)(a1 + (v11 >> 3) + 28) |= v6 )
    {
      while ( 1 )
      {
        LOBYTE(v6) = sub_A71840((__int64)v9);
        if ( !(_BYTE)v6 )
          break;
        if ( i == ++v9 )
          goto LABEL_17;
      }
      LODWORD(v6) = sub_A71AE0(v9);
      v11 = v6 + 7;
      if ( (int)v6 >= 0 )
        v11 = v6;
      ++v9;
      LODWORD(v6) = 1 << ((int)v6 % 8);
    }
LABEL_17:
    ++v3;
  }
  return v6;
}
