// Function: sub_2ACD630
// Address: 0x2acd630
//
char __fastcall sub_2ACD630(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v6; // rbx
  unsigned int v7; // r12d
  __int64 v8; // rcx
  unsigned int v9; // edi
  __int64 v10; // rsi
  __int64 *v11; // r8
  int v12; // r11d
  unsigned int v13; // edx
  __int64 *v14; // rax
  __int64 v15; // r10
  unsigned int v16; // r8d
  int v17; // r11d
  __int64 *v18; // r10
  unsigned int v19; // edx
  __int64 *v20; // rax
  __int64 v21; // r9
  int v22; // eax
  char result; // al
  unsigned int v24; // ebx
  int v25; // eax
  int v26; // edx
  __int64 v27; // rax
  __int64 v28; // rdx
  int v29; // eax
  int v30; // edx
  __int64 v31; // rax
  __int64 v32; // rdx
  unsigned int v33; // esi
  int v34; // esi
  unsigned int v35; // [rsp+Ch] [rbp-44h]
  _QWORD v36[7]; // [rsp+18h] [rbp-38h] BYREF

  v6 = *a1;
  v7 = *(_DWORD *)(*a1 + 24);
  if ( !v7 )
  {
    v36[0] = 0;
    ++*(_QWORD *)v6;
LABEL_43:
    v34 = 2 * v7;
LABEL_44:
    sub_2ACC850(v6, v34);
    sub_2AC1490(v6, (__int64 *)a2, v36);
    v30 = *(_DWORD *)(v6 + 16) + 1;
    goto LABEL_36;
  }
  v8 = *(_QWORD *)a2;
  v9 = v7 - 1;
  v10 = *(_QWORD *)(v6 + 8);
  v11 = 0;
  v12 = 1;
  v13 = (v7 - 1) & (((unsigned int)*(_QWORD *)a2 >> 9) ^ ((unsigned int)*(_QWORD *)a2 >> 4));
  v14 = (__int64 *)(v10 + 16LL * v13);
  v15 = *v14;
  if ( *(_QWORD *)a2 == *v14 )
  {
LABEL_3:
    v16 = *((_DWORD *)v14 + 2);
    goto LABEL_4;
  }
  while ( v15 != -4096 )
  {
    if ( !v11 && v15 == -8192 )
      v11 = v14;
    v13 = v9 & (v12 + v13);
    v14 = (__int64 *)(v10 + 16LL * v13);
    v15 = *v14;
    if ( v8 == *v14 )
      goto LABEL_3;
    ++v12;
  }
  if ( !v11 )
    v11 = v14;
  v36[0] = v11;
  v29 = *(_DWORD *)(v6 + 16);
  ++*(_QWORD *)v6;
  v30 = v29 + 1;
  if ( 4 * (v29 + 1) >= 3 * v7 )
    goto LABEL_43;
  if ( v7 - *(_DWORD *)(v6 + 20) - v30 <= v7 >> 3 )
  {
    v34 = v7;
    goto LABEL_44;
  }
LABEL_36:
  *(_DWORD *)(v6 + 16) = v30;
  v31 = v36[0];
  if ( *(_QWORD *)v36[0] != -4096 )
    --*(_DWORD *)(v6 + 20);
  v32 = *(_QWORD *)a2;
  *(_DWORD *)(v31 + 8) = 0;
  *(_QWORD *)v31 = v32;
  v6 = *a1;
  v7 = *(_DWORD *)(*a1 + 24);
  if ( !v7 )
  {
    v36[0] = 0;
    v33 = 0;
    ++*(_QWORD *)v6;
    goto LABEL_40;
  }
  v10 = *(_QWORD *)(v6 + 8);
  v9 = v7 - 1;
  v16 = 0;
LABEL_4:
  v17 = 1;
  v18 = 0;
  v19 = v9 & (((unsigned int)*(_QWORD *)a3 >> 9) ^ ((unsigned int)*(_QWORD *)a3 >> 4));
  v20 = (__int64 *)(v10 + 16LL * v19);
  v21 = *v20;
  if ( *v20 == *(_QWORD *)a3 )
  {
LABEL_5:
    v22 = *((_DWORD *)v20 + 2);
    goto LABEL_6;
  }
  while ( v21 != -4096 )
  {
    if ( v21 == -8192 && !v18 )
      v18 = v20;
    v19 = v9 & (v17 + v19);
    v20 = (__int64 *)(v10 + 16LL * v19);
    v21 = *v20;
    if ( *(_QWORD *)a3 == *v20 )
      goto LABEL_5;
    ++v17;
  }
  if ( !v18 )
    v18 = v20;
  v36[0] = v18;
  v25 = *(_DWORD *)(v6 + 16);
  ++*(_QWORD *)v6;
  v26 = v25 + 1;
  if ( 4 * (v25 + 1) < 3 * v7 )
  {
    if ( v7 - *(_DWORD *)(v6 + 20) - v26 <= v7 >> 3 )
    {
      v35 = v16;
      sub_2ACC850(v6, v7);
      sub_2AC1490(v6, (__int64 *)a3, v36);
      v16 = v35;
      v26 = *(_DWORD *)(v6 + 16) + 1;
    }
    goto LABEL_23;
  }
  v33 = v7;
  v7 = v16;
LABEL_40:
  sub_2ACC850(v6, 2 * v33);
  sub_2AC1490(v6, (__int64 *)a3, v36);
  v16 = v7;
  v26 = *(_DWORD *)(v6 + 16) + 1;
LABEL_23:
  *(_DWORD *)(v6 + 16) = v26;
  v27 = v36[0];
  if ( *(_QWORD *)v36[0] != -4096 )
    --*(_DWORD *)(v6 + 20);
  v28 = *(_QWORD *)a3;
  *(_DWORD *)(v27 + 8) = 0;
  *(_QWORD *)v27 = v28;
  v22 = 0;
LABEL_6:
  if ( v16 == v22 )
  {
    if ( !*(_BYTE *)(a2 + 12) )
      return *(_DWORD *)(a2 + 8) < *(_DWORD *)(a3 + 8);
    result = *(_BYTE *)(a3 + 12);
    if ( result )
      return *(_DWORD *)(a2 + 8) < *(_DWORD *)(a3 + 8);
  }
  else
  {
    v24 = *(_DWORD *)sub_2ACCA00(*a1, (__int64 *)a2);
    return v24 < *(_DWORD *)sub_2ACCA00(*a1, (__int64 *)a3);
  }
  return result;
}
