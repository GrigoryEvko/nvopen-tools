// Function: sub_29C2990
// Address: 0x29c2990
//
void __fastcall sub_29C2990(unsigned int **a1, __int64 a2, __int64 a3, unsigned __int8 a4, char a5)
{
  unsigned __int64 v7; // rcx
  char *v8; // rsi
  __int64 v9; // r14
  __int64 v10; // r13
  __int64 v11; // rax
  __int64 *v12; // r11
  __int64 v13; // r12
  __int64 v14; // r15
  unsigned int v15; // esi
  __int64 v16; // r9
  unsigned __int64 v17; // rdx
  unsigned int v18; // r8d
  __int64 *v19; // rax
  __int64 v20; // rdi
  __int64 *v21; // r10
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 *v24; // r15
  __int64 v25; // rbx
  __int64 v26; // r12
  __int64 v27; // rax
  __int64 v28; // r9
  unsigned __int64 v29; // rax
  __int64 *v30; // rcx
  int v31; // eax
  int v32; // edi
  __int8 *v33; // rsi
  __m128i *v34; // rax
  __int64 *v35; // r11
  __int64 *v36; // r10
  __int64 v37; // rax
  __int64 *v38; // r10
  unsigned __int64 v39; // rcx
  unsigned __int64 v40; // rax
  int v41; // r9d
  int v42; // r9d
  __int64 v43; // r8
  unsigned int v44; // eax
  __int64 v45; // rdx
  int v46; // esi
  __int64 *v47; // r10
  int v48; // r9d
  int v49; // r9d
  __int64 v50; // r8
  int v51; // esi
  unsigned int v52; // edx
  __int64 v53; // rax
  __int64 *v54; // [rsp+8h] [rbp-D8h]
  __int64 *v55; // [rsp+10h] [rbp-D0h]
  unsigned int v56; // [rsp+10h] [rbp-D0h]
  __int64 *v57; // [rsp+18h] [rbp-C8h]
  int v58; // [rsp+18h] [rbp-C8h]
  __int64 *v59; // [rsp+18h] [rbp-C8h]
  __int64 *v60; // [rsp+18h] [rbp-C8h]
  __int64 *v61; // [rsp+18h] [rbp-C8h]
  __int64 v65; // [rsp+40h] [rbp-A0h]
  __int64 v66[2]; // [rsp+50h] [rbp-90h] BYREF
  _QWORD v67[2]; // [rsp+60h] [rbp-80h] BYREF
  __m128i *v68; // [rsp+70h] [rbp-70h]
  __int64 v69; // [rsp+78h] [rbp-68h]
  __m128i v70; // [rsp+80h] [rbp-60h] BYREF
  __int64 v71[2]; // [rsp+90h] [rbp-50h] BYREF
  _BYTE v72[4]; // [rsp+A0h] [rbp-40h] BYREF
  char v73; // [rsp+A4h] [rbp-3Ch] BYREF
  _BYTE v74[59]; // [rsp+A5h] [rbp-3Bh] BYREF

  v7 = **a1;
  **a1 = v7 + 1;
  if ( v7 )
  {
    v8 = v74;
    do
    {
      *--v8 = v7 % 0xA + 48;
      v29 = v7;
      v7 /= 0xAu;
    }
    while ( v29 > 9 );
  }
  else
  {
    v73 = 48;
    v8 = &v73;
  }
  v9 = a2;
  v66[0] = (__int64)v67;
  sub_29C1410(v66, v8, (__int64)v74);
  if ( *(_BYTE *)(*(_QWORD *)(a2 + 8) + 8LL) == 7 )
    v9 = sub_ACD640(*(_QWORD *)a1[1], 0, 0);
  v10 = sub_B10CD0(a2 + 48);
  v57 = (__int64 *)a1[5];
  v65 = (__int64)a1[2];
  v11 = sub_29C1BE0(*v57, *(_QWORD *)(v9 + 8));
  v12 = v57;
  v13 = v11;
  v14 = v57[1];
  v15 = *(_DWORD *)(v14 + 24);
  if ( !v15 )
  {
    ++*(_QWORD *)v14;
    goto LABEL_37;
  }
  v16 = *(_QWORD *)(v14 + 8);
  v17 = ((0xBF58476D1CE4E5B9LL * v11) >> 31) ^ (0xBF58476D1CE4E5B9LL * v11);
  v18 = v17 & (v15 - 1);
  v19 = (__int64 *)(v16 + 16LL * v18);
  v20 = *v19;
  if ( v13 == *v19 )
  {
LABEL_7:
    v21 = v19 + 1;
    v22 = v19[1];
    if ( v22 )
      goto LABEL_8;
    goto LABEL_23;
  }
  v58 = 1;
  v30 = 0;
  while ( v20 != -1 )
  {
    if ( !v30 && v20 == -2 )
      v30 = v19;
    v18 = (v15 - 1) & (v58 + v18);
    v19 = (__int64 *)(v16 + 16LL * v18);
    v20 = *v19;
    if ( v13 == *v19 )
      goto LABEL_7;
    ++v58;
  }
  if ( !v30 )
    v30 = v19;
  v31 = *(_DWORD *)(v14 + 16);
  ++*(_QWORD *)v14;
  v32 = v31 + 1;
  if ( 4 * (v31 + 1) >= 3 * v15 )
  {
LABEL_37:
    v60 = v12;
    sub_29C2790(v14, 2 * v15);
    v41 = *(_DWORD *)(v14 + 24);
    if ( v41 )
    {
      v42 = v41 - 1;
      v43 = *(_QWORD *)(v14 + 8);
      v12 = v60;
      v32 = *(_DWORD *)(v14 + 16) + 1;
      v44 = v42 & (((0xBF58476D1CE4E5B9LL * v13) >> 31) ^ (484763065 * v13));
      v30 = (__int64 *)(v43 + 16LL * v44);
      v45 = *v30;
      if ( v13 == *v30 )
        goto LABEL_20;
      v46 = 1;
      v47 = 0;
      while ( v45 != -1 )
      {
        if ( v45 == -2 && !v47 )
          v47 = v30;
        v44 = v42 & (v46 + v44);
        v30 = (__int64 *)(v43 + 16LL * v44);
        v45 = *v30;
        if ( v13 == *v30 )
          goto LABEL_20;
        ++v46;
      }
LABEL_41:
      if ( v47 )
        v30 = v47;
      goto LABEL_20;
    }
LABEL_62:
    ++*(_DWORD *)(v14 + 16);
    BUG();
  }
  if ( v15 - *(_DWORD *)(v14 + 20) - v32 <= v15 >> 3 )
  {
    v56 = v17;
    v61 = v12;
    sub_29C2790(v14, v15);
    v48 = *(_DWORD *)(v14 + 24);
    if ( v48 )
    {
      v49 = v48 - 1;
      v50 = *(_QWORD *)(v14 + 8);
      v47 = 0;
      v12 = v61;
      v51 = 1;
      v32 = *(_DWORD *)(v14 + 16) + 1;
      v52 = v49 & v56;
      v30 = (__int64 *)(v50 + 16LL * (v49 & v56));
      v53 = *v30;
      if ( v13 == *v30 )
        goto LABEL_20;
      while ( v53 != -1 )
      {
        if ( v53 == -2 && !v47 )
          v47 = v30;
        v52 = v49 & (v51 + v52);
        v30 = (__int64 *)(v50 + 16LL * v52);
        v53 = *v30;
        if ( v13 == *v30 )
          goto LABEL_20;
        ++v51;
      }
      goto LABEL_41;
    }
    goto LABEL_62;
  }
LABEL_20:
  *(_DWORD *)(v14 + 16) = v32;
  if ( *v30 != -1 )
    --*(_DWORD *)(v14 + 20);
  *v30 = v13;
  v21 = v30 + 1;
  v30[1] = 0;
LABEL_23:
  if ( v13 )
  {
    v39 = v13;
    v33 = &v70.m128i_i8[5];
    do
    {
      *--v33 = v39 % 0xA + 48;
      v40 = v39;
      v39 /= 0xAu;
    }
    while ( v40 > 9 );
  }
  else
  {
    v70.m128i_i8[4] = 48;
    v33 = &v70.m128i_i8[4];
  }
  v54 = v21;
  v55 = v12;
  v71[0] = (__int64)v72;
  sub_29C1410(v71, v33, (__int64)v70.m128i_i64 + 5);
  v34 = (__m128i *)sub_2241130((unsigned __int64 *)v71, 0, 0, "ty", 2u);
  v68 = &v70;
  v35 = v55;
  v36 = v54;
  if ( (__m128i *)v34->m128i_i64[0] == &v34[1] )
  {
    v70 = _mm_loadu_si128(v34 + 1);
  }
  else
  {
    v68 = (__m128i *)v34->m128i_i64[0];
    v70.m128i_i64[0] = v34[1].m128i_i64[0];
  }
  v69 = v34->m128i_i64[1];
  v34->m128i_i64[0] = (__int64)v34[1].m128i_i64;
  v34->m128i_i64[1] = 0;
  v34[1].m128i_i8[0] = 0;
  if ( (_BYTE *)v71[0] != v72 )
  {
    j_j___libc_free_0(v71[0]);
    v36 = v54;
    v35 = v55;
  }
  v59 = v36;
  v37 = sub_ADC9A0(v35[2], (__int64)v68, v69, v13, 7, 0, 0);
  v38 = v59;
  *v59 = v37;
  if ( v68 != &v70 )
  {
    j_j___libc_free_0((unsigned __int64)v68);
    v38 = v59;
  }
  v22 = *v38;
LABEL_8:
  v23 = sub_ADFB30(v65, *(_QWORD *)a1[3], v66[0], v66[1], *(_QWORD *)a1[4], *(_DWORD *)(v10 + 4), v22, 1, 0, 0);
  v24 = (__int64 *)a1[2];
  v25 = a4;
  v26 = v23;
  BYTE1(v25) = a5;
  v27 = sub_ADD5E0((__int64)v24, 0, 0);
  sub_ADF2E0(v24, v9, v26, v27, v10, v28, a3, v25);
  if ( (_QWORD *)v66[0] != v67 )
    j_j___libc_free_0(v66[0]);
}
