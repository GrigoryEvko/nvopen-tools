// Function: sub_2E3D060
// Address: 0x2e3d060
//
void __fastcall sub_2E3D060(__int64 a1, _QWORD *a2, __int64 a3, unsigned __int64 *a4)
{
  __int64 v5; // rax
  unsigned __int64 v6; // r12
  __int64 v7; // r15
  _QWORD *v8; // rax
  _QWORD *v9; // rdx
  __int64 v10; // rax
  __int64 v11; // r9
  unsigned __int64 v12; // rdx
  unsigned __int64 v13; // r13
  __int64 v14; // r14
  char v15; // r10
  __int64 v16; // rax
  __int64 v17; // r8
  __int64 *v18; // r15
  __int64 v19; // rbx
  __int64 v20; // rax
  __int64 v21; // r12
  __int64 v22; // rcx
  __int64 v23; // rdx
  __int64 *v24; // r14
  __int64 v25; // rsi
  _QWORD *v26; // rax
  _QWORD *v27; // rax
  _QWORD *v28; // rcx
  _QWORD *v29; // rdx
  unsigned __int64 v30; // r12
  unsigned __int64 *v31; // r15
  unsigned __int64 *i; // r14
  __int64 v33; // r15
  _QWORD *v34; // r13
  __int64 v35; // rax
  __int64 v36; // rsi
  __int16 v37; // cx
  __int64 v38; // rdi
  unsigned __int64 v39; // rax
  __int64 v40; // rbx
  __int64 v41; // rax
  __int64 v42; // rcx
  unsigned int v43; // edx
  __int64 v44; // rdi
  __int64 v45; // rsi
  unsigned int v46; // ecx
  __int64 *v47; // rax
  __int64 v48; // r10
  __int64 v49; // rax
  _QWORD *v50; // r12
  __int64 v51; // r13
  __int64 v52; // rbx
  unsigned __int64 v53; // rdi
  unsigned __int64 *j; // rbx
  char v55; // dl
  unsigned int v56; // eax
  signed __int64 v57; // rax
  unsigned __int16 v58; // dx
  unsigned __int16 v59; // r12
  __int64 v60; // rdx
  unsigned __int16 v61; // dx
  unsigned __int64 v62; // rcx
  __int16 v63; // dx
  unsigned __int64 v64; // rax
  int v65; // r14d
  int v66; // eax
  int v67; // r9d
  unsigned __int64 v68; // [rsp+8h] [rbp-108h]
  unsigned __int64 *v69; // [rsp+18h] [rbp-F8h]
  unsigned __int64 v71; // [rsp+28h] [rbp-E8h]
  unsigned __int64 v72; // [rsp+30h] [rbp-E0h]
  __int64 v74; // [rsp+48h] [rbp-C8h]
  __int64 v75; // [rsp+50h] [rbp-C0h]
  __int64 v76; // [rsp+58h] [rbp-B8h]
  __int64 v77; // [rsp+58h] [rbp-B8h]
  __int64 v79; // [rsp+68h] [rbp-A8h]
  unsigned __int64 v80; // [rsp+68h] [rbp-A8h]
  __int64 v81; // [rsp+70h] [rbp-A0h]
  _QWORD *v82; // [rsp+70h] [rbp-A0h]
  __int64 *v83; // [rsp+78h] [rbp-98h]
  __int64 v84; // [rsp+78h] [rbp-98h]
  unsigned __int16 v85; // [rsp+84h] [rbp-8Ch] BYREF
  unsigned __int16 v86; // [rsp+86h] [rbp-8Ah] BYREF
  unsigned __int64 v87; // [rsp+88h] [rbp-88h] BYREF
  __m128i v88; // [rsp+90h] [rbp-80h] BYREF
  unsigned __int16 v89; // [rsp+A0h] [rbp-70h]
  __m128i v90; // [rsp+B0h] [rbp-60h] BYREF
  __int64 v91; // [rsp+C0h] [rbp-50h]
  int v92; // [rsp+C8h] [rbp-48h]
  char v93; // [rsp+CCh] [rbp-44h]
  char v94; // [rsp+D0h] [rbp-40h] BYREF

  v5 = a2[1] - *a2;
  v75 = v5 >> 3;
  if ( (unsigned __int64)v5 > 0x2AAAAAAAAAAAAAA8LL )
    sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
  v6 = 16 * v75;
  v68 = 3 * v75;
  v7 = 3 * v75;
  if ( !v75 )
  {
    v28 = 0;
    v27 = 0;
    v71 = 0;
    v69 = 0;
    v72 = 0;
    goto LABEL_30;
  }
  v8 = (_QWORD *)sub_22077B0(24 * v75);
  v72 = (unsigned __int64)v8;
  v9 = &v8[v7];
  v69 = &v8[v7];
  do
  {
    if ( v8 )
    {
      *v8 = 0;
      v8[1] = 0;
      v8[2] = 0;
    }
    v8 += 3;
  }
  while ( v8 != v9 );
  v10 = sub_22077B0(v6);
  v12 = v10 + v6;
  v71 = v10;
  do
  {
    if ( v10 )
    {
      *(_QWORD *)v10 = 0;
      *(_WORD *)(v10 + 8) = 0;
    }
    v10 += 16;
  }
  while ( v10 != v12 );
  v13 = v71;
  v14 = a3;
  v81 = 0;
  v74 = v72;
  do
  {
    v15 = 1;
    v16 = *(_QWORD *)(*a2 + 8 * v81);
    v93 = 1;
    v90.m128i_i64[0] = 0;
    v90.m128i_i64[1] = (__int64)&v94;
    v91 = 2;
    v92 = 0;
    v17 = *(_QWORD *)(v16 + 112);
    v79 = v16;
    v18 = (__int64 *)v17;
    v83 = (__int64 *)(v17 + 8LL * *(unsigned int *)(v16 + 120));
    if ( v83 == (__int64 *)v17 )
      goto LABEL_25;
    v19 = v14;
    do
    {
      while ( 1 )
      {
        v20 = *(unsigned int *)(v19 + 24);
        v21 = *v18;
        v22 = *(_QWORD *)(v19 + 8);
        if ( (_DWORD)v20 )
        {
          v23 = ((_DWORD)v20 - 1) & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
          v24 = (__int64 *)(v22 + 16 * v23);
          v25 = *v24;
          if ( v21 != *v24 )
          {
            v65 = 1;
            while ( v25 != -4096 )
            {
              v17 = (unsigned int)(v65 + 1);
              v23 = ((_DWORD)v20 - 1) & (unsigned int)(v65 + v23);
              v24 = (__int64 *)(v22 + 16LL * (unsigned int)v23);
              v25 = *v24;
              if ( v21 == *v24 )
                goto LABEL_16;
              v65 = v17;
            }
            goto LABEL_22;
          }
LABEL_16:
          if ( v24 != (__int64 *)(v22 + 16 * v20) )
            break;
        }
LABEL_22:
        if ( v83 == ++v18 )
          goto LABEL_23;
      }
      if ( !v15 )
        goto LABEL_57;
      v26 = (_QWORD *)v90.m128i_i64[1];
      v22 = HIDWORD(v91);
      v23 = v90.m128i_i64[1] + 8LL * HIDWORD(v91);
      if ( v90.m128i_i64[1] != v23 )
      {
        while ( v21 != *v26 )
        {
          if ( (_QWORD *)v23 == ++v26 )
            goto LABEL_65;
        }
        goto LABEL_22;
      }
LABEL_65:
      if ( HIDWORD(v91) >= (unsigned int)v91 )
      {
LABEL_57:
        sub_C8CC70((__int64)&v90, *v18, v23, v22, v17, v11);
        v15 = v93;
        if ( v55 )
          goto LABEL_58;
        goto LABEL_22;
      }
      ++HIDWORD(v91);
      *(_QWORD *)v23 = v21;
      ++v90.m128i_i64[0];
LABEL_58:
      v56 = sub_2E441D0(*(_QWORD *)(a1 + 112), v79, v21);
      if ( v56 )
      {
        v57 = sub_F04200(v56, 0x80000000);
        v59 = v58;
        v60 = v24[1];
        v88.m128i_i64[1] = v57;
        v77 = v57;
        v88.m128i_i64[0] = v60;
        v89 = v59;
        sub_2E3D020(v74, &v88);
        v61 = *(_WORD *)(v13 + 8);
        v62 = *(_QWORD *)v13;
        v86 = v59;
        v87 = v62;
        v85 = v61;
        v88.m128i_i64[0] = v77;
        v63 = sub_FDCA70(&v87, &v85, (unsigned __int64 *)&v88, &v86);
        v64 = v87 + v88.m128i_i64[0];
        if ( __CFADD__(v87, v88.m128i_i64[0]) )
        {
          ++v63;
          v64 = (v64 >> 1) | 0x8000000000000000LL;
        }
        *(_QWORD *)v13 = v64;
        *(_WORD *)(v13 + 8) = v63;
        if ( v63 > 0x3FFF )
        {
          *(_QWORD *)v13 = -1;
          *(_WORD *)(v13 + 8) = 0x3FFF;
        }
      }
      v15 = v93;
      ++v18;
    }
    while ( v83 != v18 );
LABEL_23:
    v14 = v19;
    if ( !v15 )
      _libc_free(v90.m128i_u64[1]);
LABEL_25:
    ++v81;
    v13 += 16LL;
    v74 += 24;
  }
  while ( v81 != v75 );
  a3 = v14;
  v27 = (_QWORD *)sub_22077B0(v68 * 8);
  v28 = v27;
  v29 = &v27[v68];
  do
  {
    if ( v27 )
    {
      *v27 = 0;
      v27[1] = 0;
      v27[2] = 0;
    }
    v27 += 3;
  }
  while ( v29 != v27 );
LABEL_30:
  v30 = *a4;
  v31 = (unsigned __int64 *)a4[1];
  *a4 = (unsigned __int64)v28;
  a4[1] = (unsigned __int64)v27;
  a4[2] = (unsigned __int64)v27;
  for ( i = (unsigned __int64 *)v30; v31 != i; i += 3 )
  {
    if ( *i )
      j_j___libc_free_0(*i);
  }
  if ( v30 )
    j_j___libc_free_0(v30);
  v84 = 0;
  v33 = v71;
  v80 = v72;
  if ( v75 )
  {
    v76 = a3;
    do
    {
      v34 = *(_QWORD **)v80;
      v82 = *(_QWORD **)(v80 + 8);
      if ( v82 != *(_QWORD **)v80 )
      {
        do
        {
          v35 = *v34;
          v36 = v34[1];
          v34 += 3;
          v37 = *((_WORD *)v34 - 4);
          v38 = 3 * v35;
          v39 = *a4;
          v90.m128i_i64[0] = v36;
          v90.m128i_i16[4] = v37;
          v40 = v39 + 8 * v38;
          v41 = sub_FDE760((__int64)&v90, v33);
          v42 = *(_QWORD *)v41;
          LOWORD(v41) = *(_WORD *)(v41 + 8);
          v90.m128i_i64[0] = v84;
          v90.m128i_i64[1] = v42;
          LOWORD(v91) = v41;
          sub_2E3D020(v40, &v90);
        }
        while ( v82 != v34 );
      }
      ++v84;
      v33 += 16;
      v80 += 24LL;
    }
    while ( v84 != v75 );
    a3 = v76;
  }
  v43 = *(_DWORD *)(a3 + 24);
  v44 = *(_QWORD *)(a3 + 8);
  v45 = *(_QWORD *)(*(_QWORD *)(a1 + 128) + 328LL);
  if ( v43 )
  {
    v46 = (v43 - 1) & (((unsigned int)v45 >> 9) ^ ((unsigned int)v45 >> 4));
    v47 = (__int64 *)(v44 + 16LL * v46);
    v48 = *v47;
    if ( v45 == *v47 )
      goto LABEL_44;
    v66 = 1;
    while ( v48 != -4096 )
    {
      v67 = v66 + 1;
      v46 = (v43 - 1) & (v66 + v46);
      v47 = (__int64 *)(v44 + 16LL * v46);
      v48 = *v47;
      if ( v45 == *v47 )
        goto LABEL_44;
      v66 = v67;
    }
  }
  v47 = (__int64 *)(v44 + 16LL * v43);
LABEL_44:
  v49 = v47[1];
  if ( v75 )
  {
    v50 = (_QWORD *)v72;
    v51 = 0;
    v52 = 24 * v49;
    do
    {
      while ( v50[1] != *v50 )
      {
        ++v51;
        v50 += 3;
        if ( v51 == v75 )
          goto LABEL_49;
      }
      v90.m128i_i64[0] = v51++;
      v90.m128i_i64[1] = 1;
      v50 += 3;
      v53 = *a4;
      LOWORD(v91) = 0;
      sub_2E3D020(v52 + v53, &v90);
    }
    while ( v51 != v75 );
  }
LABEL_49:
  if ( v71 )
    j_j___libc_free_0(v71);
  for ( j = (unsigned __int64 *)v72; j != v69; j += 3 )
  {
    if ( *j )
      j_j___libc_free_0(*j);
  }
  if ( v72 )
    j_j___libc_free_0(v72);
}
