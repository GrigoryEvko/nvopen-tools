// Function: sub_ADFB30
// Address: 0xadfb30
//
__int64 __fastcall sub_ADFB30(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        int a6,
        __int64 a7,
        char a8,
        int a9,
        int a10)
{
  int v11; // r13d
  __int64 v13; // rax
  unsigned int v14; // r8d
  __int64 v15; // rcx
  __int64 v16; // r12
  __int64 v17; // r10
  unsigned int v18; // edx
  unsigned int v19; // r9d
  __int64 *v20; // rsi
  __int64 v21; // rdi
  __int64 v22; // rsi
  __int64 *v24; // rax
  int v25; // edi
  int v26; // edi
  int v27; // edx
  int v28; // edx
  __int64 v29; // r8
  unsigned int v30; // r10d
  __int64 v31; // rsi
  int v32; // r9d
  __int64 *v33; // r11
  int v34; // esi
  int v35; // esi
  __int64 v36; // r8
  int v37; // r9d
  unsigned int v38; // r10d
  __int64 v39; // rdx
  __int64 v40; // [rsp+8h] [rbp-58h]
  __int64 v41; // [rsp+10h] [rbp-50h]
  int v43; // [rsp+18h] [rbp-48h]
  __int64 v44; // [rsp+18h] [rbp-48h]
  unsigned int v45; // [rsp+18h] [rbp-48h]

  v11 = a2;
  v41 = a1 + 400;
  v13 = sub_AF34D0(a2);
  v14 = *(_DWORD *)(a1 + 424);
  v15 = a3;
  v16 = v13;
  if ( !v14 )
  {
    ++*(_QWORD *)(a1 + 400);
    goto LABEL_15;
  }
  v17 = *(_QWORD *)(a1 + 408);
  v18 = ((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4);
  v19 = (v14 - 1) & v18;
  v20 = (__int64 *)(v17 + 56LL * v19);
  v21 = *v20;
  if ( v13 == *v20 )
  {
LABEL_3:
    v22 = (__int64)(v20 + 1);
    return sub_ADDBB0(*(_QWORD *)(a1 + 8), v22, v11, v15, a4, 0, a5, a6, a7, a8, a9, a10, 0);
  }
  v43 = 1;
  v24 = 0;
  while ( v21 != -4096 )
  {
    if ( !v24 && v21 == -8192 )
      v24 = v20;
    v19 = (v14 - 1) & (v43 + v19);
    v20 = (__int64 *)(v17 + 56LL * v19);
    v21 = *v20;
    if ( v16 == *v20 )
      goto LABEL_3;
    ++v43;
  }
  v25 = *(_DWORD *)(a1 + 416);
  if ( !v24 )
    v24 = v20;
  ++*(_QWORD *)(a1 + 400);
  v26 = v25 + 1;
  if ( 4 * v26 >= 3 * v14 )
  {
LABEL_15:
    v44 = v15;
    sub_ADF3C0(v41, 2 * v14);
    v27 = *(_DWORD *)(a1 + 424);
    if ( v27 )
    {
      v28 = v27 - 1;
      v29 = *(_QWORD *)(a1 + 408);
      v30 = v28 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
      v26 = *(_DWORD *)(a1 + 416) + 1;
      v15 = v44;
      v24 = (__int64 *)(v29 + 56LL * v30);
      v31 = *v24;
      if ( v16 == *v24 )
        goto LABEL_11;
      v32 = 1;
      v33 = 0;
      while ( v31 != -4096 )
      {
        if ( !v33 && v31 == -8192 )
          v33 = v24;
        v30 = v28 & (v32 + v30);
        v24 = (__int64 *)(v29 + 56LL * v30);
        v31 = *v24;
        if ( v16 == *v24 )
          goto LABEL_11;
        ++v32;
      }
LABEL_19:
      if ( v33 )
        v24 = v33;
      goto LABEL_11;
    }
LABEL_40:
    ++*(_DWORD *)(a1 + 416);
    BUG();
  }
  if ( v14 - *(_DWORD *)(a1 + 420) - v26 <= v14 >> 3 )
  {
    v40 = v15;
    v45 = v18;
    sub_ADF3C0(v41, v14);
    v34 = *(_DWORD *)(a1 + 424);
    if ( v34 )
    {
      v35 = v34 - 1;
      v36 = *(_QWORD *)(a1 + 408);
      v33 = 0;
      v37 = 1;
      v38 = v35 & v45;
      v26 = *(_DWORD *)(a1 + 416) + 1;
      v15 = v40;
      v24 = (__int64 *)(v36 + 56LL * (v35 & v45));
      v39 = *v24;
      if ( v16 == *v24 )
        goto LABEL_11;
      while ( v39 != -4096 )
      {
        if ( !v33 && v39 == -8192 )
          v33 = v24;
        v38 = v35 & (v37 + v38);
        v24 = (__int64 *)(v36 + 56LL * v38);
        v39 = *v24;
        if ( v16 == *v24 )
          goto LABEL_11;
        ++v37;
      }
      goto LABEL_19;
    }
    goto LABEL_40;
  }
LABEL_11:
  *(_DWORD *)(a1 + 416) = v26;
  if ( *v24 != -4096 )
    --*(_DWORD *)(a1 + 420);
  *v24 = v16;
  v22 = (__int64)(v24 + 1);
  v24[1] = (__int64)(v24 + 3);
  v24[2] = 0x400000000LL;
  return sub_ADDBB0(*(_QWORD *)(a1 + 8), v22, v11, v15, a4, 0, a5, a6, a7, a8, a9, a10, 0);
}
