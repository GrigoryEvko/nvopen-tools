// Function: sub_103B1D0
// Address: 0x103b1d0
//
_BOOL8 __fastcall sub_103B1D0(__int64 a1, __int64 a2)
{
  unsigned __int8 *v2; // r13
  unsigned __int8 *v3; // rdx
  int v5; // edx
  __int64 v6; // r14
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // r15
  int v10; // r15d
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  unsigned __int8 *v14; // r13
  int v15; // edx
  __int64 v16; // r14
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // r15
  int v20; // r15d
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rdx
  unsigned __int8 *v24; // r13
  int v25; // edx
  _QWORD *v26; // rbx
  __int64 v27; // r15
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // r14
  int v31; // r14d
  __int64 v32; // rax
  __int64 v33; // rdx
  unsigned __int8 *v34; // rax
  __int64 v35; // [rsp-40h] [rbp-40h]

  if ( *(_BYTE *)a1 != *(_BYTE *)a2 )
    return 0;
  v2 = *(unsigned __int8 **)(a1 + 8);
  v3 = *(unsigned __int8 **)(a2 + 8);
  if ( !*(_BYTE *)a1 )
    return v2 == v3
        && *(_QWORD *)(a1 + 16) == *(_QWORD *)(a2 + 16)
        && *(_QWORD *)(a1 + 24) == *(_QWORD *)(a2 + 24)
        && *(_QWORD *)(a1 + 32) == *(_QWORD *)(a2 + 32)
        && *(_QWORD *)(a1 + 40) == *(_QWORD *)(a2 + 40)
        && *(_QWORD *)(a1 + 48) == *(_QWORD *)(a2 + 48);
  if ( *((_QWORD *)v2 - 4) != *((_QWORD *)v3 - 4) )
    return 0;
  v5 = *v2;
  if ( v5 == 40 )
  {
    v6 = 32LL * (unsigned int)sub_B491D0(*(_QWORD *)(a1 + 8));
  }
  else
  {
    v6 = 0;
    if ( v5 != 85 )
    {
      v6 = 64;
      if ( v5 != 34 )
LABEL_63:
        BUG();
    }
  }
  if ( (v2[7] & 0x80u) == 0 )
    goto LABEL_25;
  v7 = sub_BD2BC0((__int64)v2);
  v9 = v7 + v8;
  if ( (v2[7] & 0x80u) == 0 )
  {
    if ( (unsigned int)(v9 >> 4) )
LABEL_67:
      BUG();
LABEL_25:
    v13 = 0;
    goto LABEL_26;
  }
  if ( !(unsigned int)((v9 - sub_BD2BC0((__int64)v2)) >> 4) )
    goto LABEL_25;
  if ( (v2[7] & 0x80u) == 0 )
    goto LABEL_67;
  v10 = *(_DWORD *)(sub_BD2BC0((__int64)v2) + 8);
  if ( (v2[7] & 0x80u) == 0 )
    BUG();
  v11 = sub_BD2BC0((__int64)v2);
  v13 = 32LL * (unsigned int)(*(_DWORD *)(v11 + v12 - 4) - v10);
LABEL_26:
  v35 = (32LL * (*((_DWORD *)v2 + 1) & 0x7FFFFFF) - 32 - v6 - v13) >> 5;
  v14 = *(unsigned __int8 **)(a2 + 8);
  v15 = *v14;
  if ( v15 == 40 )
  {
    v16 = 32LL * (unsigned int)sub_B491D0(*(_QWORD *)(a2 + 8));
  }
  else
  {
    v16 = 0;
    if ( v15 != 85 )
    {
      v16 = 64;
      if ( v15 != 34 )
        goto LABEL_63;
    }
  }
  if ( (v14[7] & 0x80u) == 0 )
    goto LABEL_37;
  v17 = sub_BD2BC0((__int64)v14);
  v19 = v17 + v18;
  if ( (v14[7] & 0x80u) == 0 )
  {
    if ( (unsigned int)(v19 >> 4) )
LABEL_64:
      BUG();
LABEL_37:
    v23 = 0;
    goto LABEL_38;
  }
  if ( !(unsigned int)((v19 - sub_BD2BC0((__int64)v14)) >> 4) )
    goto LABEL_37;
  if ( (v14[7] & 0x80u) == 0 )
    goto LABEL_64;
  v20 = *(_DWORD *)(sub_BD2BC0((__int64)v14) + 8);
  if ( (v14[7] & 0x80u) == 0 )
    BUG();
  v21 = sub_BD2BC0((__int64)v14);
  v23 = 32LL * (unsigned int)(*(_DWORD *)(v21 + v22 - 4) - v20);
LABEL_38:
  if ( (_DWORD)v35 != (unsigned int)((32LL * (*((_DWORD *)v14 + 1) & 0x7FFFFFF) - 32 - v16 - v23) >> 5) )
    return 0;
  v24 = *(unsigned __int8 **)(a1 + 8);
  v25 = *v24;
  v26 = (_QWORD *)(*(_QWORD *)(a2 + 8) - 32LL * (*(_DWORD *)(*(_QWORD *)(a2 + 8) + 4LL) & 0x7FFFFFF));
  if ( v25 == 40 )
  {
    v27 = -32 - 32LL * (unsigned int)sub_B491D0(*(_QWORD *)(a1 + 8));
  }
  else
  {
    v27 = -32;
    if ( v25 != 85 )
    {
      v27 = -96;
      if ( v25 != 34 )
        goto LABEL_63;
    }
  }
  if ( (v24[7] & 0x80u) != 0 )
  {
    v28 = sub_BD2BC0((__int64)v24);
    v30 = v28 + v29;
    if ( (v24[7] & 0x80u) == 0 )
    {
      if ( (unsigned int)(v30 >> 4) )
        goto LABEL_61;
    }
    else if ( (unsigned int)((v30 - sub_BD2BC0((__int64)v24)) >> 4) )
    {
      if ( (v24[7] & 0x80u) != 0 )
      {
        v31 = *(_DWORD *)(sub_BD2BC0((__int64)v24) + 8);
        if ( (v24[7] & 0x80u) == 0 )
          BUG();
        v32 = sub_BD2BC0((__int64)v24);
        v27 -= 32LL * (unsigned int)(*(_DWORD *)(v32 + v33 - 4) - v31);
        goto LABEL_52;
      }
LABEL_61:
      BUG();
    }
  }
LABEL_52:
  v34 = (unsigned __int8 *)(*(_QWORD *)(a1 + 8) - 32LL * (*(_DWORD *)(*(_QWORD *)(a1 + 8) + 4LL) & 0x7FFFFFF));
  if ( v34 == &v24[v27] )
    return 1;
  while ( *(_QWORD *)v34 == *v26 )
  {
    v34 += 32;
    v26 += 4;
    if ( v34 == &v24[v27] )
      return 1;
  }
  return 0;
}
