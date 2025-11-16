// Function: sub_E027A0
// Address: 0xe027a0
//
__int64 __fastcall sub_E027A0(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r14
  __int64 v8; // rsi
  char v9; // al
  __int64 v10; // rax
  unsigned __int64 v11; // rdx
  __int64 v12; // r15
  char v13; // al
  __int64 v14; // rax
  __int64 v15; // rbx
  __int64 result; // rax
  char v17; // r15
  __int64 v18; // rax
  __int64 v19; // rdx
  unsigned __int64 v20; // r9
  unsigned __int64 v21; // rax
  __int64 v22; // rcx
  __int64 v23; // rbx
  __int16 v24; // ax
  __int64 v25; // r15
  __int64 v26; // rax
  unsigned int v27; // r12d
  __int64 v28; // [rsp+0h] [rbp-50h]
  __int64 v29; // [rsp+0h] [rbp-50h]
  unsigned __int64 v31; // [rsp+10h] [rbp-40h] BYREF
  __int64 v32; // [rsp+18h] [rbp-38h]

  v4 = a3 + 312;
  while ( 1 )
  {
    while ( 1 )
    {
      if ( *(_BYTE *)a1 == 6 )
        a1 = *(_QWORD *)(a1 - 32);
      v8 = *(_QWORD *)(a1 + 8);
      if ( *(_BYTE *)(v8 + 8) == 14 )
      {
        result = a1;
        if ( a2 )
          return 0;
        return result;
      }
      v9 = *(_BYTE *)a1;
      if ( *(_BYTE *)a1 == 10 )
      {
        v10 = sub_AE4AC0(v4, v8);
        v11 = *(_QWORD *)v10;
        v12 = v10;
        LOBYTE(v10) = *(_BYTE *)(v10 + 8);
        v31 = v11;
        LOBYTE(v32) = v10;
        if ( sub_CA1930(&v31) <= a2 )
          return 0;
        v28 = (unsigned int)sub_AE1C80(v12, a2);
        v13 = *(_BYTE *)(v12 + 16 * v28 + 32);
        v31 = *(_QWORD *)(v12 + 16 * v28 + 24);
        LOBYTE(v32) = v13;
        v14 = sub_CA1930(&v31);
        if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
          v15 = *(_QWORD *)(a1 - 8);
        else
          v15 = a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
        return sub_E027A0(*(_QWORD *)(v15 + 32 * v28), a2 - v14, a3, a4);
      }
      if ( v9 != 9 )
        break;
      v29 = *(_QWORD *)(v8 + 24);
      v17 = sub_AE5020(v4, v29);
      v18 = sub_9208B0(v4, v29);
      v32 = v19;
      v31 = ((1LL << v17) + ((unsigned __int64)(v18 + 7) >> 3) - 1) >> v17 << v17;
      v20 = sub_CA1930(&v31);
      v21 = a2 / v20;
      v22 = *(_DWORD *)(a1 + 4) & 0x7FFFFFF;
      if ( (unsigned int)v22 <= (unsigned int)(a2 / v20) )
        return 0;
      a2 %= v20;
      if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
        v23 = *(_QWORD *)(a1 - 8);
      else
        v23 = a1 - 32 * v22;
      a1 = *(_QWORD *)(v23 + 32LL * (unsigned int)v21);
    }
    if ( v9 == 17 && !a2 )
      break;
    if ( v9 != 5 )
      return 0;
    v24 = *(_WORD *)(a1 + 2);
    if ( v24 == 38 || v24 == 47 )
    {
      a1 = *(_QWORD *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
    }
    else
    {
      if ( v24 != 15 )
        return 0;
      v25 = *(_QWORD *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
      v26 = sub_E027A0(*(_QWORD *)(a1 + 32 * (1LL - (*(_DWORD *)(a1 + 4) & 0x7FFFFFF))), 0, a3, 0);
      if ( *(_BYTE *)v26 == 5 && *(_WORD *)(v26 + 2) == 34 )
        v26 = *(_QWORD *)(v26 - 32LL * (*(_DWORD *)(v26 + 4) & 0x7FFFFFF));
      if ( a4 != v26 )
        return 0;
      a1 = v25;
    }
  }
  v27 = *(_DWORD *)(a1 + 32);
  if ( v27 > 0x40 )
  {
    if ( v27 != (unsigned int)sub_C444A0(a1 + 24) )
      return 0;
  }
  else if ( *(_QWORD *)(a1 + 24) )
  {
    return 0;
  }
  return a1;
}
