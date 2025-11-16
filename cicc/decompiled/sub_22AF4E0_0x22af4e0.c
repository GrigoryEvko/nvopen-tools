// Function: sub_22AF4E0
// Address: 0x22af4e0
//
__int64 __fastcall sub_22AF4E0(__int64 a1, __int64 a2)
{
  __int64 v3; // r14
  __int64 v4; // r15
  unsigned int v5; // r13d
  char v6; // al
  int v7; // r14d
  int v8; // eax
  unsigned int v9; // ecx
  __int64 v10; // rax
  __int64 v11; // rsi
  __int64 v12; // rdx
  __int64 v13; // rdi
  bool v14; // bl
  _QWORD *v16; // rax
  _QWORD *i; // rdx
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // r14
  const void *v21; // r15
  const void *v22; // rax
  size_t v23; // rdx
  unsigned __int8 v24; // [rsp+Fh] [rbp-31h]

  if ( !*(_BYTE *)(a1 + 72) )
    return 0;
  v24 = *(_BYTE *)(a2 + 72);
  if ( !v24 )
    return 0;
  v3 = *(_QWORD *)(a1 + 16);
  v4 = *(_QWORD *)(a2 + 16);
  v5 = sub_B46250(v3, v4, 0);
  v6 = *(_BYTE *)v3;
  if ( !(_BYTE)v5 )
  {
    if ( (unsigned __int8)(v6 - 82) <= 1u && (unsigned __int8)(*(_BYTE *)v4 - 82) <= 1u )
    {
      v7 = sub_22AF4B0(a1);
      v8 = sub_22AF4B0(a2);
      v9 = v24;
      if ( v7 == v8 )
      {
        v10 = *(_QWORD *)(a1 + 24);
        v11 = v10 + 8LL * *(unsigned int *)(a1 + 32);
        if ( v10 != v11 )
        {
          v12 = *(_QWORD *)(a2 + 24);
          v13 = v12 + 8LL * *(unsigned int *)(a2 + 32);
          while ( v13 != v12 && *(_QWORD *)(*(_QWORD *)v10 + 8LL) == *(_QWORD *)(*(_QWORD *)v12 + 8LL) )
          {
            v10 += 8;
            v12 += 8;
            if ( v11 == v10 )
              return v9;
          }
          LOBYTE(v9) = v13 == v12;
        }
        return v9;
      }
      return v5;
    }
    return 0;
  }
  if ( v6 != 63 )
  {
    if ( v6 == 85 )
    {
      if ( *(_BYTE *)v4 != 85 )
        return v5;
      v18 = sub_22AF4D0(a2);
      v20 = v19;
      v21 = (const void *)v18;
      v22 = (const void *)sub_22AF4D0(a1);
      if ( v23 != v20 || v23 && memcmp(v22, v21, v23) )
        return 0;
      v6 = **(_BYTE **)(a1 + 16);
    }
    if ( v6 == 31 && **(_BYTE **)(a2 + 16) == 31 )
      LOBYTE(v5) = *(_DWORD *)(a1 + 136) == *(_DWORD *)(a2 + 136);
    return v5;
  }
  v14 = sub_B4DE30(v3);
  if ( v14 != sub_B4DE30(v4) )
    return 0;
  v16 = (_QWORD *)(v3 + 32 * (2LL - (*(_DWORD *)(v3 + 4) & 0x7FFFFFF)));
  if ( (_QWORD *)v3 != v16 )
  {
    for ( i = (_QWORD *)(v4 + 32 * (2LL - (*(_DWORD *)(v4 + 4) & 0x7FFFFFF))); (_QWORD *)v4 != i && *v16 == *i; i += 4 )
    {
      v16 += 4;
      if ( (_QWORD *)v3 == v16 )
        return v5;
    }
    LOBYTE(v5) = v4 == (_QWORD)i;
  }
  return v5;
}
