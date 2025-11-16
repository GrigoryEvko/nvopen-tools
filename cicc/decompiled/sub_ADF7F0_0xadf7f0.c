// Function: sub_ADF7F0
// Address: 0xadf7f0
//
__int64 __fastcall sub_ADF7F0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        int a5,
        __int64 a6,
        int a7,
        __int64 a8,
        char a9,
        int a10,
        __int64 a11)
{
  int v12; // r13d
  __int64 v14; // rax
  unsigned int v15; // r8d
  __int64 v16; // rcx
  __int64 v17; // r12
  __int64 v18; // r10
  unsigned int v19; // edx
  unsigned int v20; // r9d
  __int64 *v21; // rsi
  __int64 v22; // rdi
  __int64 v23; // rsi
  __int64 *v25; // rax
  int v26; // edi
  int v27; // edi
  int v28; // edx
  int v29; // edx
  __int64 v30; // r8
  unsigned int v31; // r10d
  __int64 v32; // rsi
  int v33; // r9d
  __int64 *v34; // r11
  int v35; // esi
  int v36; // esi
  __int64 v37; // r8
  int v38; // r9d
  unsigned int v39; // r10d
  __int64 v40; // rdx
  __int64 v41; // [rsp+8h] [rbp-58h]
  __int64 v42; // [rsp+10h] [rbp-50h]
  int v44; // [rsp+18h] [rbp-48h]
  __int64 v45; // [rsp+18h] [rbp-48h]
  unsigned int v46; // [rsp+18h] [rbp-48h]

  v12 = a2;
  v42 = a1 + 400;
  v14 = sub_AF34D0(a2);
  v15 = *(_DWORD *)(a1 + 424);
  v16 = a3;
  v17 = v14;
  if ( !v15 )
  {
    ++*(_QWORD *)(a1 + 400);
    goto LABEL_15;
  }
  v18 = *(_QWORD *)(a1 + 408);
  v19 = ((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4);
  v20 = (v15 - 1) & v19;
  v21 = (__int64 *)(v18 + 56LL * v20);
  v22 = *v21;
  if ( v14 == *v21 )
  {
LABEL_3:
    v23 = (__int64)(v21 + 1);
    return sub_ADDBB0(*(_QWORD *)(a1 + 8), v23, v12, v16, a4, a5, a6, a7, a8, a9, a10, 0, a11);
  }
  v44 = 1;
  v25 = 0;
  while ( v22 != -4096 )
  {
    if ( !v25 && v22 == -8192 )
      v25 = v21;
    v20 = (v15 - 1) & (v44 + v20);
    v21 = (__int64 *)(v18 + 56LL * v20);
    v22 = *v21;
    if ( v17 == *v21 )
      goto LABEL_3;
    ++v44;
  }
  v26 = *(_DWORD *)(a1 + 416);
  if ( !v25 )
    v25 = v21;
  ++*(_QWORD *)(a1 + 400);
  v27 = v26 + 1;
  if ( 4 * v27 >= 3 * v15 )
  {
LABEL_15:
    v45 = v16;
    sub_ADF3C0(v42, 2 * v15);
    v28 = *(_DWORD *)(a1 + 424);
    if ( v28 )
    {
      v29 = v28 - 1;
      v30 = *(_QWORD *)(a1 + 408);
      v31 = v29 & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
      v27 = *(_DWORD *)(a1 + 416) + 1;
      v16 = v45;
      v25 = (__int64 *)(v30 + 56LL * v31);
      v32 = *v25;
      if ( v17 == *v25 )
        goto LABEL_11;
      v33 = 1;
      v34 = 0;
      while ( v32 != -4096 )
      {
        if ( !v34 && v32 == -8192 )
          v34 = v25;
        v31 = v29 & (v33 + v31);
        v25 = (__int64 *)(v30 + 56LL * v31);
        v32 = *v25;
        if ( v17 == *v25 )
          goto LABEL_11;
        ++v33;
      }
LABEL_19:
      if ( v34 )
        v25 = v34;
      goto LABEL_11;
    }
LABEL_40:
    ++*(_DWORD *)(a1 + 416);
    BUG();
  }
  if ( v15 - *(_DWORD *)(a1 + 420) - v27 <= v15 >> 3 )
  {
    v41 = v16;
    v46 = v19;
    sub_ADF3C0(v42, v15);
    v35 = *(_DWORD *)(a1 + 424);
    if ( v35 )
    {
      v36 = v35 - 1;
      v37 = *(_QWORD *)(a1 + 408);
      v34 = 0;
      v38 = 1;
      v39 = v36 & v46;
      v27 = *(_DWORD *)(a1 + 416) + 1;
      v16 = v41;
      v25 = (__int64 *)(v37 + 56LL * (v36 & v46));
      v40 = *v25;
      if ( v17 == *v25 )
        goto LABEL_11;
      while ( v40 != -4096 )
      {
        if ( !v34 && v40 == -8192 )
          v34 = v25;
        v39 = v36 & (v38 + v39);
        v25 = (__int64 *)(v37 + 56LL * v39);
        v40 = *v25;
        if ( v17 == *v25 )
          goto LABEL_11;
        ++v38;
      }
      goto LABEL_19;
    }
    goto LABEL_40;
  }
LABEL_11:
  *(_DWORD *)(a1 + 416) = v27;
  if ( *v25 != -4096 )
    --*(_DWORD *)(a1 + 420);
  *v25 = v17;
  v23 = (__int64)(v25 + 1);
  v25[1] = (__int64)(v25 + 3);
  v25[2] = 0x400000000LL;
  return sub_ADDBB0(*(_QWORD *)(a1 + 8), v23, v12, v16, a4, a5, a6, a7, a8, a9, a10, 0, a11);
}
