// Function: sub_2A5AE50
// Address: 0x2a5ae50
//
__int64 __fastcall sub_2A5AE50(__int64 *a1, __int64 **a2)
{
  __int64 *v3; // r8
  __int64 *i; // rsi
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 *v8; // rcx
  __int64 *j; // rax
  __int64 v10; // rdx
  unsigned __int64 v11; // rcx
  __int64 *v12; // r12
  __int64 *v13; // r10
  __int64 *v14; // rax
  __int64 v15; // rsi
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // r8
  double v19; // xmm2_8
  unsigned __int64 v20; // rdx
  double v21; // xmm0_8
  __int64 v22; // rdx
  __int64 v23; // rdx
  double v24; // xmm1_8
  __int64 *v25; // r11
  __int64 *v26; // r12
  __int64 v27; // r13
  __int64 v28; // rsi
  __int64 v29; // rax
  __int64 v30; // r9
  __int64 **v31; // rdx
  unsigned __int64 v32; // r8
  __int64 *v33; // r10
  __int64 *v34; // rcx
  unsigned __int64 v35; // rax
  __int64 v36; // r13
  unsigned __int64 v37; // rdx
  __int64 *v38; // r11
  __int64 *v39; // r9
  __int64 v40; // r10
  __int64 v41; // rax
  __int64 **v42; // rcx
  __int64 *v43; // rdx
  __int64 *v44; // r8
  __int64 v45; // r9
  __int64 v46; // rcx
  __int64 v47; // rsi
  unsigned __int64 v48; // rax
  __int64 v49; // r14
  unsigned int v50; // r10d
  __int64 *v51; // rdx
  __int64 v52; // rax
  __int64 v53; // r8
  __int64 v54; // rsi
  __int64 v55; // rcx
  unsigned __int64 v57; // r13

  v3 = a2[1];
  for ( i = *a2; v3 != i; ++i )
  {
    v5 = *i;
    v6 = 72 * *i;
    *(_QWORD *)(*a1 + v6 + 32) = 0;
    *(_QWORD *)(*a1 + v6 + 40) = 0;
    v7 = a1[8] + 24 * v5;
    v8 = *(__int64 **)(v7 + 8);
    for ( j = *(__int64 **)v7; v8 != j; *(_QWORD *)(v10 + 48) = 0 )
      v10 = *j++;
  }
  v11 = 0x4000000000000LL;
  *(_QWORD *)(*a1 + 72 * a1[6] + 32) = 0x3FF0000000000000LL;
  v12 = a2[1];
  v13 = *a2;
  if ( *a2 != v12 )
  {
    do
    {
      v14 = (__int64 *)(a1[8] + 24 * *v13);
      v15 = v14[1];
      v16 = *v14;
      v17 = (v15 - v16) >> 3;
      if ( v16 != v15 )
      {
        v18 = 72 * *v13;
        if ( v17 < 0 )
        {
          v57 = ((v15 - v16) >> 3) & 1 | ((unsigned __int64)((v15 - v16) >> 3) >> 1);
          v19 = (double)(int)v57 + (double)(int)v57;
        }
        else
        {
          v19 = (double)(int)v17;
        }
        do
        {
          v21 = *(double *)(*a1 + v18 + 32) / v19;
          v22 = *a1 + 72LL * *(_QWORD *)(*(_QWORD *)v16 + 24LL);
          *(double *)(v22 + 32) = *(double *)(v22 + 32) + v21;
          v23 = *(_QWORD *)(*(_QWORD *)v16 + 8LL);
          if ( v23 != 0x4000000000000LL )
          {
            v24 = (double)((int)v23 - *(_DWORD *)(*(_QWORD *)v16 + 16LL)) / v21;
            v20 = v24 < 9.223372036854776e18
                ? (unsigned int)(int)v24
                : (unsigned int)(int)(v24 - 9.223372036854776e18) ^ 0x8000000000000000LL;
            if ( v11 > v20 )
              v11 = v20;
          }
          v16 += 8;
        }
        while ( v15 != v16 );
      }
      ++v13;
    }
    while ( v12 != v13 );
    if ( !v11 )
      return 0;
  }
  *(_QWORD *)(*a1 + 72 * a1[6] + 40) = v11;
  v25 = *a2;
  v26 = a2[1];
  v27 = a1[7];
  v28 = *a1;
  if ( v26 != *a2 )
  {
    do
    {
      v29 = *v25;
      if ( *v25 == v27 )
        break;
      v30 = 72 * v29;
      v31 = (__int64 **)(a1[8] + 24 * v29);
      v32 = *(_QWORD *)(v28 + 72 * v29 + 40);
      v33 = v31[1];
      v34 = *v31;
      v35 = (v32 + v33 - *v31 - 1) / (v33 - *v31);
      if ( v33 != *v31 )
      {
        while ( 1 )
        {
          v36 = *v34;
          v37 = *(_QWORD *)(*v34 + 8) - *(_QWORD *)(*v34 + 16);
          if ( v37 > v35 )
            v37 = v35;
          if ( v37 > v32 )
            v37 = v32;
          ++v34;
          *(_QWORD *)(v28 + 72LL * *(_QWORD *)(v36 + 24) + 40) += v37;
          *(_QWORD *)(v30 + *a1 + 40) -= v37;
          *(_QWORD *)(*(v34 - 1) + 48) += v37;
          if ( v33 == v34 )
            break;
          v28 = *a1;
          v32 = *(_QWORD *)(*a1 + v30 + 40);
        }
        v27 = a1[7];
        v28 = *a1;
      }
      ++v25;
    }
    while ( v26 != v25 );
  }
  *(_QWORD *)(v28 + 72 * v27 + 40) = 0;
  v38 = a2[1];
  v39 = *a2;
  v40 = v38 - *a2 - 1;
  if ( v38 - *a2 != 1 )
  {
    do
    {
      v41 = v39[--v40];
      v42 = (__int64 **)(a1[8] + 24 * v41);
      v43 = *v42;
      v44 = v42[1];
      if ( v44 != *v42 )
      {
        v45 = 72 * v41;
        do
        {
          v46 = *v43;
          v47 = *a1 + 72LL * *(_QWORD *)(*v43 + 24);
          v48 = *(_QWORD *)(v47 + 40);
          if ( v48 )
          {
            v49 = *(_QWORD *)(v47 + 40);
            if ( *(_QWORD *)(v46 + 48) <= v48 )
              v49 = *(_QWORD *)(v46 + 48);
            *(_QWORD *)(v47 + 40) = v48 - v49;
            *(_QWORD *)(v45 + *a1 + 40) += v49;
            *(_QWORD *)(*v43 + 48) -= v49;
          }
          ++v43;
        }
        while ( v44 != v43 );
        v39 = *a2;
      }
    }
    while ( v40 );
    v38 = a2[1];
  }
  if ( v38 == v39 )
    return 0;
  v50 = 0;
  do
  {
    v51 = (__int64 *)(a1[8] + 24 * *v39);
    v52 = *v51;
    v53 = v51[1];
    if ( v53 != *v51 )
    {
      do
      {
        v54 = 7LL * *(_QWORD *)(*(_QWORD *)v52 + 32LL);
        v55 = *(_QWORD *)(a1[3] + 24LL * *(_QWORD *)(*(_QWORD *)v52 + 24LL));
        *(_QWORD *)(*(_QWORD *)v52 + 16LL) += *(_QWORD *)(*(_QWORD *)v52 + 48LL);
        *(_QWORD *)(v55 + 8 * v54 + 16) -= *(_QWORD *)(*(_QWORD *)v52 + 48LL);
        if ( *(_QWORD *)(*(_QWORD *)v52 + 8LL) == *(_QWORD *)(*(_QWORD *)v52 + 16LL) )
        {
          if ( *(_QWORD *)(*(_QWORD *)v52 + 48LL) )
            v50 = 1;
        }
        v52 += 8;
      }
      while ( v53 != v52 );
    }
    ++v39;
  }
  while ( v38 != v39 );
  return v50;
}
