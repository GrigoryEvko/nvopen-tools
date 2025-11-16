// Function: sub_16707E0
// Address: 0x16707e0
//
_BOOL8 __fastcall sub_16707E0(__int64 a1, __int64 a2)
{
  _BOOL4 v3; // r15d
  int v4; // eax
  int v5; // ecx
  __int64 v6; // rdi
  int v7; // r9d
  unsigned int v8; // eax
  __int64 v9; // rsi
  int v11; // ebx
  int v12; // ebx
  __int64 *v13; // r12
  int v14; // r12d
  __int64 v15; // [rsp+0h] [rbp-50h]
  __int64 v16; // [rsp+8h] [rbp-48h]
  int v17; // [rsp+10h] [rbp-40h]
  unsigned int i; // [rsp+14h] [rbp-3Ch]
  __int64 v19; // [rsp+18h] [rbp-38h]

  v3 = (*(_DWORD *)(a2 + 8) & 0x100) == 0;
  if ( (*(_DWORD *)(a2 + 8) & 0x100) == 0 )
  {
    v4 = *(_DWORD *)(a1 + 24);
    if ( v4 )
    {
      v5 = v4 - 1;
      v6 = *(_QWORD *)(a1 + 8);
      v7 = 1;
      v8 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v9 = *(_QWORD *)(v6 + 8LL * v8);
      if ( a2 == v9 )
        return v3;
      while ( v9 != -8 )
      {
        v8 = v5 & (v7 + v8);
        v9 = *(_QWORD *)(v6 + 8LL * v8);
        if ( a2 == v9 )
          return v3;
        ++v7;
      }
    }
    return 0;
  }
  v11 = *(_DWORD *)(a1 + 56);
  v19 = *(_QWORD *)(a1 + 40);
  if ( v11 )
  {
    v12 = v11 - 1;
    v16 = sub_16704E0();
    v15 = sub_16704F0();
    v17 = 1;
    for ( i = v12 & sub_16707B0(a2); ; i = v14 )
    {
      v13 = (__int64 *)(v19 + 8LL * i);
      if ( sub_1670560(a2, *v13) )
        break;
      if ( sub_1670560(*v13, v16) )
        return v3;
      sub_1670560(*v13, v15);
      v14 = v12 & (v17 + i);
      ++v17;
    }
    if ( v13 != (__int64 *)(*(_QWORD *)(a1 + 40) + 8LL * *(unsigned int *)(a1 + 56)) )
      LOBYTE(v3) = *v13 == a2;
  }
  return v3;
}
