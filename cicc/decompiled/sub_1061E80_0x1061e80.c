// Function: sub_1061E80
// Address: 0x1061e80
//
_BOOL8 __fastcall sub_1061E80(__int64 a1, __int64 a2)
{
  _BOOL4 v3; // r15d
  int v4; // eax
  __int64 v5; // rsi
  int v6; // ecx
  int v7; // r9d
  unsigned int v8; // eax
  __int64 v9; // rdi
  int v11; // ebx
  int v12; // ebx
  __int64 *v13; // r12
  int v14; // r12d
  __int64 v15; // [rsp+8h] [rbp-48h]
  int v16; // [rsp+10h] [rbp-40h]
  unsigned int i; // [rsp+14h] [rbp-3Ch]
  __int64 v18; // [rsp+18h] [rbp-38h]

  v3 = (*(_DWORD *)(a2 + 8) & 0x100) == 0;
  if ( (*(_DWORD *)(a2 + 8) & 0x100) == 0 )
  {
    v4 = *(_DWORD *)(a1 + 24);
    v5 = *(_QWORD *)(a1 + 8);
    if ( v4 )
    {
      v6 = v4 - 1;
      v7 = 1;
      v8 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v9 = *(_QWORD *)(v5 + 8LL * v8);
      if ( a2 == v9 )
        return v3;
      while ( v9 != -4096 )
      {
        v8 = v6 & (v7 + v8);
        v9 = *(_QWORD *)(v5 + 8LL * v8);
        if ( a2 == v9 )
          return v3;
        ++v7;
      }
    }
    return 0;
  }
  v11 = *(_DWORD *)(a1 + 56);
  v18 = *(_QWORD *)(a1 + 40);
  if ( v11 )
  {
    v12 = v11 - 1;
    v15 = sub_1061AC0();
    v16 = 1;
    for ( i = v12 & sub_1061E50(a2); ; i = v14 )
    {
      v13 = (__int64 *)(v18 + 8LL * i);
      if ( sub_1061B40(a2, *v13) )
        break;
      if ( sub_1061B40(*v13, v15) )
        return v3;
      v14 = v12 & (v16 + i);
      ++v16;
    }
    if ( v13 != (__int64 *)(*(_QWORD *)(a1 + 40) + 8LL * *(unsigned int *)(a1 + 56)) )
      LOBYTE(v3) = *v13 == a2;
  }
  return v3;
}
