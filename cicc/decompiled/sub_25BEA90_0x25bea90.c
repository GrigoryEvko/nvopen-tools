// Function: sub_25BEA90
// Address: 0x25bea90
//
__int64 __fastcall sub_25BEA90(__int64 a1, __int64 a2, char a3)
{
  unsigned __int8 *v5; // r14
  int v6; // eax
  unsigned __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // r15
  char v10; // al
  char v11; // al
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // rax
  _QWORD *v18; // rdi
  _QWORD *v19; // rsi
  int v20; // edx
  __int64 v21; // r12
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rdx
  __int64 v27; // r12
  __int64 v28; // rax
  int v29; // eax
  int v30; // edx
  unsigned int v31; // eax
  __int64 v32; // rdi
  __int64 v33; // [rsp+0h] [rbp-50h]
  int v34; // [rsp+0h] [rbp-50h]
  __int64 v35; // [rsp+8h] [rbp-48h]
  __int64 v36[7]; // [rsp+18h] [rbp-38h] BYREF

  v5 = *(unsigned __int8 **)(a2 + 24);
  v6 = *v5;
  if ( (unsigned __int8)v6 <= 0x1Cu )
    goto LABEL_7;
  v7 = (unsigned int)(v6 - 34);
  if ( (unsigned __int8)(v6 - 34) > 0x33u )
  {
    if ( (_BYTE)v6 == 30 )
    {
      *(_BYTE *)(a1 + 9) |= a3;
      v11 = *(_BYTE *)(a1 + 8);
      return v11 != 15;
    }
    goto LABEL_7;
  }
  v8 = 0x8000000000041LL;
  if ( !_bittest64(&v8, v7) )
    goto LABEL_7;
  v9 = *((_QWORD *)v5 - 4);
  if ( !v9 )
    goto LABEL_7;
  if ( *(_BYTE *)v9 )
    goto LABEL_7;
  if ( *(_QWORD *)(v9 + 24) != *((_QWORD *)v5 + 10) )
    goto LABEL_7;
  v36[0] = *((_QWORD *)v5 - 4);
  if ( sub_B2FC80(v36[0]) || (unsigned __int8)sub_B2FC00((_BYTE *)v9) )
    goto LABEL_7;
  v17 = *(_QWORD *)(a1 + 64);
  if ( !*(_DWORD *)(v17 + 16) )
  {
    v18 = *(_QWORD **)(v17 + 32);
    v19 = &v18[*(unsigned int *)(v17 + 40)];
    if ( v19 != sub_25BD100(v18, (__int64)v19, v36) )
      goto LABEL_15;
LABEL_7:
    v10 = *(_BYTE *)(a1 + 8);
    *(_BYTE *)(a1 + 9) |= a3;
    v11 = a3 | v10;
    *(_BYTE *)(a1 + 8) = v11;
    return v11 != 15;
  }
  v19 = *(_QWORD **)(v17 + 8);
  v29 = *(_DWORD *)(v17 + 24);
  if ( !v29 )
    goto LABEL_7;
  v30 = v29 - 1;
  v31 = (v29 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
  v32 = v19[v31];
  if ( v9 != v32 )
  {
    v15 = 1;
    while ( v32 != -4096 )
    {
      v16 = (unsigned int)(v15 + 1);
      v31 = v30 & (v15 + v31);
      v32 = v19[v31];
      if ( v9 == v32 )
        goto LABEL_15;
      v15 = (unsigned int)v16;
    }
    goto LABEL_7;
  }
LABEL_15:
  v20 = *v5;
  v35 = (a2 - (__int64)&v5[-32 * (*((_DWORD *)v5 + 1) & 0x7FFFFFF)]) >> 5;
  if ( v20 == 40 )
  {
    v19 = (_QWORD *)(32LL * (unsigned int)sub_B491D0((__int64)v5));
    v21 = (__int64)v19;
  }
  else
  {
    v21 = 0;
    if ( v20 != 85 )
    {
      v21 = 64;
      if ( v20 != 34 )
        BUG();
    }
  }
  if ( (v5[7] & 0x80u) == 0 )
    goto LABEL_39;
  v22 = sub_BD2BC0((__int64)v5);
  v33 = v23 + v22;
  if ( (v5[7] & 0x80u) == 0 )
  {
    if ( (unsigned int)(v33 >> 4) )
LABEL_41:
      BUG();
LABEL_39:
    v26 = 0;
    goto LABEL_23;
  }
  if ( !(unsigned int)((v33 - sub_BD2BC0((__int64)v5)) >> 4) )
    goto LABEL_39;
  if ( (v5[7] & 0x80u) == 0 )
    goto LABEL_41;
  v34 = *(_DWORD *)(sub_BD2BC0((__int64)v5) + 8);
  if ( (v5[7] & 0x80u) == 0 )
    BUG();
  v24 = sub_BD2BC0((__int64)v5);
  v26 = 32LL * (unsigned int)(*(_DWORD *)(v24 + v25 - 4) - v34);
LABEL_23:
  if ( (unsigned int)v35 >= (unsigned int)((32LL * (*((_DWORD *)v5 + 1) & 0x7FFFFFF) - 32 - v21 - v26) >> 5)
    || (unsigned __int64)(unsigned int)v35 >= *(_QWORD *)(v9 + 104) )
  {
    goto LABEL_7;
  }
  if ( (*(_BYTE *)(v9 + 2) & 1) != 0 )
    sub_B2C6D0(v9, (__int64)v19, v26, v14);
  v27 = *(_QWORD *)(v9 + 96) + 40LL * (unsigned int)v35;
  v28 = *(unsigned int *)(a1 + 24);
  if ( v28 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 28) )
  {
    sub_C8D5F0(a1 + 16, (const void *)(a1 + 32), v28 + 1, 8u, v15, v16);
    v28 = *(unsigned int *)(a1 + 24);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8 * v28) = v27;
  ++*(_DWORD *)(a1 + 24);
  return 2;
}
