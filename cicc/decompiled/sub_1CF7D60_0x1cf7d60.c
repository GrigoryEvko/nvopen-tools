// Function: sub_1CF7D60
// Address: 0x1cf7d60
//
__int64 ***__fastcall sub_1CF7D60(
        __int64 a1,
        __m128 a2,
        double a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        __m128 a9)
{
  __int64 v9; // rax
  __int64 v11; // r12
  __int64 v12; // r13
  unsigned int v13; // esi
  __int64 v14; // rdi
  unsigned int v15; // ecx
  _QWORD *v16; // rdx
  __int64 v17; // rax
  __int64 ***v18; // r12
  __int64 v19; // rax
  double v20; // xmm4_8
  double v21; // xmm5_8
  __int64 ***v22; // r12
  __int64 v23; // rbx
  unsigned int v24; // esi
  __int64 v25; // rdi
  unsigned int v26; // ecx
  __int64 ****v27; // rdx
  __int64 ***result; // rax
  int v29; // r10d
  __int64 ****v30; // r9
  int v31; // eax
  int v32; // edx
  int v33; // r10d
  _QWORD *v34; // r9
  int v35; // eax
  int v36; // edx
  int v37; // eax
  int v38; // ecx
  __int64 v39; // rdi
  __int64 ***v40; // rsi
  int v41; // r10d
  __int64 ****v42; // r8
  int v43; // eax
  __int64 v44; // rsi
  int v45; // r8d
  __int64 ****v46; // rdi
  unsigned int v47; // r13d
  __int64 ***v48; // rcx
  int v49; // eax
  int v50; // ecx
  __int64 v51; // rdi
  unsigned int v52; // eax
  __int64 v53; // rsi
  int v54; // r10d
  _QWORD *v55; // r8
  int v56; // eax
  int v57; // eax
  __int64 v58; // rsi
  int v59; // r8d
  _QWORD *v60; // rdi
  unsigned int v61; // r14d
  __int64 v62; // rcx

  v9 = *(_QWORD *)(a1 + 16);
  if ( !v9 )
    goto LABEL_5;
  v11 = *(_QWORD *)(v9 + 24);
  if ( !v11 )
    goto LABEL_5;
  v12 = *(_QWORD *)(a1 + 1320);
  v13 = *(_DWORD *)(v12 + 24);
  if ( !v13 )
  {
    ++*(_QWORD *)v12;
    goto LABEL_41;
  }
  v14 = *(_QWORD *)(v12 + 8);
  v15 = (v13 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
  v16 = (_QWORD *)(v14 + 8LL * v15);
  v17 = *v16;
  if ( v11 == *v16 )
    goto LABEL_5;
  v33 = 1;
  v34 = 0;
  while ( v17 != -8 )
  {
    if ( v34 || v17 != -16 )
      v16 = v34;
    v15 = (v13 - 1) & (v33 + v15);
    v17 = *(_QWORD *)(v14 + 8LL * v15);
    if ( v11 == v17 )
      goto LABEL_5;
    ++v33;
    v34 = v16;
    v16 = (_QWORD *)(v14 + 8LL * v15);
  }
  v35 = *(_DWORD *)(v12 + 16);
  if ( !v34 )
    v34 = v16;
  ++*(_QWORD *)v12;
  v36 = v35 + 1;
  if ( 4 * (v35 + 1) >= 3 * v13 )
  {
LABEL_41:
    sub_1467110(v12, 2 * v13);
    v49 = *(_DWORD *)(v12 + 24);
    if ( v49 )
    {
      v50 = v49 - 1;
      v51 = *(_QWORD *)(v12 + 8);
      v52 = (v49 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
      v34 = (_QWORD *)(v51 + 8LL * v52);
      v53 = *v34;
      v36 = *(_DWORD *)(v12 + 16) + 1;
      if ( v11 != *v34 )
      {
        v54 = 1;
        v55 = 0;
        while ( v53 != -8 )
        {
          if ( v53 == -16 && !v55 )
            v55 = v34;
          v52 = v50 & (v54 + v52);
          v34 = (_QWORD *)(v51 + 8LL * v52);
          v53 = *v34;
          if ( v11 == *v34 )
            goto LABEL_23;
          ++v54;
        }
        if ( v55 )
          v34 = v55;
      }
      goto LABEL_23;
    }
    goto LABEL_85;
  }
  if ( v13 - *(_DWORD *)(v12 + 20) - v36 <= v13 >> 3 )
  {
    sub_1467110(v12, v13);
    v56 = *(_DWORD *)(v12 + 24);
    if ( v56 )
    {
      v57 = v56 - 1;
      v58 = *(_QWORD *)(v12 + 8);
      v59 = 1;
      v60 = 0;
      v61 = v57 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
      v34 = (_QWORD *)(v58 + 8LL * v61);
      v62 = *v34;
      v36 = *(_DWORD *)(v12 + 16) + 1;
      if ( v11 != *v34 )
      {
        while ( v62 != -8 )
        {
          if ( v62 == -16 && !v60 )
            v60 = v34;
          v61 = v57 & (v59 + v61);
          v34 = (_QWORD *)(v58 + 8LL * v61);
          v62 = *v34;
          if ( v11 == *v34 )
            goto LABEL_23;
          ++v59;
        }
        if ( v60 )
          v34 = v60;
      }
      goto LABEL_23;
    }
LABEL_85:
    ++*(_DWORD *)(v12 + 16);
    BUG();
  }
LABEL_23:
  *(_DWORD *)(v12 + 16) = v36;
  if ( *v34 != -8 )
    --*(_DWORD *)(v12 + 20);
  *v34 = v11;
LABEL_5:
  v18 = *(__int64 ****)a1;
  v19 = sub_1599EF0(**(__int64 ****)a1);
  sub_164D160((__int64)v18, v19, a2, a3, a4, a5, v20, v21, a8, a9);
  v22 = *(__int64 ****)a1;
  v23 = *(_QWORD *)(a1 + 1320);
  v24 = *(_DWORD *)(v23 + 24);
  if ( !v24 )
  {
    ++*(_QWORD *)v23;
    goto LABEL_27;
  }
  v25 = *(_QWORD *)(v23 + 8);
  v26 = (v24 - 1) & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
  v27 = (__int64 ****)(v25 + 8LL * v26);
  result = *v27;
  if ( v22 == *v27 )
    return result;
  v29 = 1;
  v30 = 0;
  while ( result != (__int64 ***)-8LL )
  {
    if ( v30 || result != (__int64 ***)-16LL )
      v27 = v30;
    v26 = (v24 - 1) & (v29 + v26);
    result = *(__int64 ****)(v25 + 8LL * v26);
    if ( v22 == result )
      return result;
    ++v29;
    v30 = v27;
    v27 = (__int64 ****)(v25 + 8LL * v26);
  }
  v31 = *(_DWORD *)(v23 + 16);
  if ( !v30 )
    v30 = v27;
  ++*(_QWORD *)v23;
  v32 = v31 + 1;
  if ( 4 * (v31 + 1) >= 3 * v24 )
  {
LABEL_27:
    sub_1467110(v23, 2 * v24);
    v37 = *(_DWORD *)(v23 + 24);
    if ( v37 )
    {
      v38 = v37 - 1;
      v39 = *(_QWORD *)(v23 + 8);
      result = (__int64 ***)((v37 - 1) & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4)));
      v30 = (__int64 ****)(v39 + 8LL * (_QWORD)result);
      v40 = *v30;
      v32 = *(_DWORD *)(v23 + 16) + 1;
      if ( v22 != *v30 )
      {
        v41 = 1;
        v42 = 0;
        while ( v40 != (__int64 ***)-8LL )
        {
          if ( !v42 && v40 == (__int64 ***)-16LL )
            v42 = v30;
          result = (__int64 ***)(v38 & (unsigned int)(v41 + (_DWORD)result));
          v30 = (__int64 ****)(v39 + 8LL * (unsigned int)result);
          v40 = *v30;
          if ( v22 == *v30 )
            goto LABEL_14;
          ++v41;
        }
        if ( v42 )
          v30 = v42;
      }
      goto LABEL_14;
    }
    goto LABEL_84;
  }
  result = (__int64 ***)(v24 - *(_DWORD *)(v23 + 20) - v32);
  if ( (unsigned int)result <= v24 >> 3 )
  {
    sub_1467110(v23, v24);
    v43 = *(_DWORD *)(v23 + 24);
    if ( v43 )
    {
      result = (__int64 ***)(unsigned int)(v43 - 1);
      v44 = *(_QWORD *)(v23 + 8);
      v45 = 1;
      v46 = 0;
      v47 = (unsigned int)result & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
      v30 = (__int64 ****)(v44 + 8LL * v47);
      v48 = *v30;
      v32 = *(_DWORD *)(v23 + 16) + 1;
      if ( v22 != *v30 )
      {
        while ( v48 != (__int64 ***)-8LL )
        {
          if ( v48 == (__int64 ***)-16LL && !v46 )
            v46 = v30;
          v47 = (unsigned int)result & (v45 + v47);
          v30 = (__int64 ****)(v44 + 8LL * v47);
          v48 = *v30;
          if ( v22 == *v30 )
            goto LABEL_14;
          ++v45;
        }
        if ( v46 )
          v30 = v46;
      }
      goto LABEL_14;
    }
LABEL_84:
    ++*(_DWORD *)(v23 + 16);
    BUG();
  }
LABEL_14:
  *(_DWORD *)(v23 + 16) = v32;
  if ( *v30 != (__int64 ***)-8LL )
    --*(_DWORD *)(v23 + 20);
  *v30 = v22;
  return result;
}
