// Function: sub_38E3D90
// Address: 0x38e3d90
//
void __fastcall sub_38E3D90(unsigned __int64 *a1, unsigned __int64 *a2, __int64 a3)
{
  unsigned __int64 *v3; // r8
  unsigned __int64 v5; // rcx
  unsigned __int64 v6; // rbx
  unsigned __int64 v7; // r14
  __int64 v8; // r13
  unsigned __int64 v9; // r15
  __int64 v10; // rsi
  unsigned __int64 v11; // rdi
  __int64 v12; // rbx
  unsigned __int64 v13; // r8
  __int64 v14; // rdx
  bool v15; // cc
  __int64 v16; // rdi
  __int64 v17; // rsi
  unsigned __int64 v18; // rax
  unsigned __int64 v19; // rdi
  __int64 v20; // r13
  __int64 v21; // rax
  __int64 v22; // r14
  __int64 v23; // r15
  unsigned int v24; // eax
  const void **v25; // rsi
  unsigned __int64 v26; // rbx
  unsigned __int64 v27; // r15
  unsigned __int64 v28; // rdi
  unsigned __int64 v29; // rdx
  __int64 v30; // r14
  __int64 v31; // rbx
  __int64 v32; // rsi
  __int64 v33; // rax
  unsigned __int64 v34; // rdi
  unsigned __int64 v35; // rbx
  unsigned int v36; // edx
  __int64 v37; // rsi
  __int64 v38; // rax
  unsigned __int64 v39; // [rsp-48h] [rbp-48h]
  unsigned __int64 *v40; // [rsp-48h] [rbp-48h]
  __int64 v41; // [rsp-40h] [rbp-40h]
  unsigned __int64 v42; // [rsp-40h] [rbp-40h]
  unsigned __int64 v43; // [rsp-40h] [rbp-40h]
  unsigned __int64 v44; // [rsp-40h] [rbp-40h]
  unsigned __int64 v45; // [rsp-40h] [rbp-40h]

  if ( a2 == a1 )
    return;
  v3 = a2;
  v5 = a2[1];
  v6 = *a2;
  v7 = *a1;
  v8 = v5 - *a2;
  if ( a1[2] - *a1 < v8 )
  {
    if ( v8 )
    {
      if ( (unsigned __int64)v8 > 0x7FFFFFFFFFFFFFF8LL )
        sub_4261EA(a1, a2, a3);
      v42 = a2[1];
      v21 = sub_22077B0(v42 - *a2);
      v5 = v42;
      v22 = v21;
    }
    else
    {
      v22 = 0;
    }
    v23 = v22;
    if ( v5 == v6 )
    {
LABEL_30:
      v26 = a1[1];
      v27 = *a1;
      if ( v26 != *a1 )
      {
        do
        {
          if ( *(_DWORD *)(v27 + 32) > 0x40u )
          {
            v28 = *(_QWORD *)(v27 + 24);
            if ( v28 )
              j_j___libc_free_0_0(v28);
          }
          v27 += 40LL;
        }
        while ( v26 != v27 );
        v27 = *a1;
      }
      if ( v27 )
        j_j___libc_free_0(v27);
      v20 = v22 + v8;
      *a1 = v22;
      a1[2] = v20;
      goto LABEL_19;
    }
    while ( 1 )
    {
      if ( !v23 )
        goto LABEL_26;
      *(_DWORD *)v23 = *(_DWORD *)v6;
      *(__m128i *)(v23 + 8) = _mm_loadu_si128((const __m128i *)(v6 + 8));
      v24 = *(_DWORD *)(v6 + 32);
      *(_DWORD *)(v23 + 32) = v24;
      if ( v24 <= 0x40 )
      {
        *(_QWORD *)(v23 + 24) = *(_QWORD *)(v6 + 24);
LABEL_26:
        v6 += 40LL;
        v23 += 40;
        if ( v5 == v6 )
          goto LABEL_30;
      }
      else
      {
        v25 = (const void **)(v6 + 24);
        v43 = v5;
        v6 += 40LL;
        sub_16A4FD0(v23 + 24, v25);
        v5 = v43;
        v23 += 40;
        if ( v43 == v6 )
          goto LABEL_30;
      }
    }
  }
  v9 = a1[1];
  v10 = v9 - v7;
  v11 = v9 - v7;
  if ( v8 <= v9 - v7 )
  {
    if ( v8 <= 0 )
      goto LABEL_17;
    v12 = v6 + 24;
    v13 = 0xCCCCCCCCCCCCCCCDLL * (v8 >> 3);
    v14 = v7 + 24;
    while ( 1 )
    {
      while ( 1 )
      {
        v15 = *(_DWORD *)(v14 + 8) <= 0x40u;
        *(_DWORD *)(v14 - 24) = *(_DWORD *)(v12 - 24);
        *(__m128i *)(v14 - 16) = _mm_loadu_si128((const __m128i *)(v12 - 16));
        if ( v15 && *(_DWORD *)(v12 + 8) <= 0x40u )
          break;
        v39 = v13;
        v41 = v14;
        sub_16A51C0(v14, v12);
        v13 = v39;
        v14 = v41;
LABEL_7:
        v14 += 40;
        v12 += 40;
        if ( !--v13 )
          goto LABEL_12;
      }
      v16 = *(_QWORD *)v12;
      *(_QWORD *)v14 = *(_QWORD *)v12;
      v17 = *(unsigned int *)(v12 + 8);
      *(_DWORD *)(v14 + 8) = v17;
      v18 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v17;
      if ( (unsigned int)v17 > 0x40 )
      {
        v37 = (unsigned int)((unsigned __int64)(v17 + 63) >> 6) - 1;
        *(_QWORD *)(v16 + 8 * v37) &= v18;
        goto LABEL_7;
      }
      v14 += 40;
      v12 += 40;
      *(_QWORD *)(v14 - 40) = v16 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v17);
      if ( !--v13 )
      {
LABEL_12:
        v7 += v8;
        while ( v9 != v7 )
        {
          if ( *(_DWORD *)(v7 + 32) > 0x40u )
          {
            v19 = *(_QWORD *)(v7 + 24);
            if ( v19 )
              j_j___libc_free_0_0(v19);
          }
          v7 += 40LL;
LABEL_17:
          ;
        }
LABEL_18:
        v20 = *a1 + v8;
        goto LABEL_19;
      }
    }
  }
  v29 = 0xCCCCCCCCCCCCCCCDLL * (v10 >> 3);
  if ( v10 > 0 )
  {
    v30 = v7 + 24;
    v31 = v6 + 24;
    while ( 1 )
    {
      while ( 1 )
      {
        v15 = *(_DWORD *)(v30 + 8) <= 0x40u;
        *(_DWORD *)(v30 - 24) = *(_DWORD *)(v31 - 24);
        *(__m128i *)(v30 - 16) = _mm_loadu_si128((const __m128i *)(v31 - 16));
        if ( v15 && *(_DWORD *)(v31 + 8) <= 0x40u )
          break;
        v40 = v3;
        v44 = v29;
        sub_16A51C0(v30, v31);
        v3 = v40;
        v29 = v44;
LABEL_43:
        v30 += 40;
        v31 += 40;
        if ( !--v29 )
          goto LABEL_48;
      }
      v32 = *(_QWORD *)v31;
      *(_QWORD *)v30 = *(_QWORD *)v31;
      v33 = *(unsigned int *)(v31 + 8);
      *(_DWORD *)(v30 + 8) = v33;
      v34 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v33;
      if ( (unsigned int)v33 > 0x40 )
      {
        v38 = (unsigned int)((unsigned __int64)(v33 + 63) >> 6) - 1;
        *(_QWORD *)(v32 + 8 * v38) &= v34;
        goto LABEL_43;
      }
      v30 += 40;
      v31 += 40;
      *(_QWORD *)(v30 - 40) = v32 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v33);
      if ( !--v29 )
      {
LABEL_48:
        v9 = a1[1];
        v7 = *a1;
        v5 = v3[1];
        v6 = *v3;
        v11 = v9 - *a1;
        break;
      }
    }
  }
  v35 = v11 + v6;
  if ( v5 != v35 )
  {
    do
    {
      if ( v9 )
      {
        *(_DWORD *)v9 = *(_DWORD *)v35;
        *(__m128i *)(v9 + 8) = _mm_loadu_si128((const __m128i *)(v35 + 8));
        v36 = *(_DWORD *)(v35 + 32);
        *(_DWORD *)(v9 + 32) = v36;
        if ( v36 <= 0x40 )
        {
          *(_QWORD *)(v9 + 24) = *(_QWORD *)(v35 + 24);
        }
        else
        {
          v45 = v5;
          sub_16A4FD0(v9 + 24, (const void **)(v35 + 24));
          v5 = v45;
        }
      }
      v35 += 40LL;
      v9 += 40LL;
    }
    while ( v5 != v35 );
    goto LABEL_18;
  }
  v20 = v7 + v8;
LABEL_19:
  a1[1] = v20;
}
