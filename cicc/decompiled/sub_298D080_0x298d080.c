// Function: sub_298D080
// Address: 0x298d080
//
__int64 __fastcall sub_298D080(_QWORD *a1)
{
  _QWORD *v2; // rdx
  __int64 v3; // r9
  __int64 v4; // rcx
  __int64 v5; // r9
  __int64 v6; // rax
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  unsigned __int64 v10; // rax
  unsigned __int64 *v11; // rdi
  unsigned __int64 *v12; // rsi
  __int64 v13; // r8
  __int64 v14; // r9
  const __m128i *v15; // rcx
  const __m128i *v16; // rdx
  unsigned __int64 v17; // r12
  __m128i *v18; // rax
  __int64 v19; // rcx
  const __m128i *v20; // rax
  const __m128i *v21; // rcx
  unsigned __int64 v22; // rbx
  __int64 v23; // rax
  unsigned __int64 v24; // rdi
  __m128i *v25; // rdx
  unsigned __int64 v26; // rax
  unsigned __int64 v27; // rcx
  __int64 v28; // rbx
  __int64 v29; // r12
  __int64 v30; // r13
  __int64 v31; // rdx
  __int64 v32; // rsi
  unsigned __int64 v33; // rax
  char v34; // si
  __int64 *v36; // [rsp+18h] [rbp-278h]
  __int64 i; // [rsp+18h] [rbp-278h]
  __int64 *v38; // [rsp+20h] [rbp-270h]
  __int64 v39; // [rsp+20h] [rbp-270h]
  char v40; // [rsp+28h] [rbp-268h]
  __int64 v41; // [rsp+28h] [rbp-268h]
  __int64 v42[8]; // [rsp+30h] [rbp-260h] BYREF
  __int64 v43; // [rsp+70h] [rbp-220h] BYREF
  _BYTE *v44; // [rsp+78h] [rbp-218h]
  __int64 v45; // [rsp+80h] [rbp-210h]
  int v46; // [rsp+88h] [rbp-208h]
  char v47; // [rsp+8Ch] [rbp-204h]
  _BYTE v48[64]; // [rsp+90h] [rbp-200h] BYREF
  __m128i *v49; // [rsp+D0h] [rbp-1C0h] BYREF
  __int64 v50; // [rsp+D8h] [rbp-1B8h]
  __int8 *v51; // [rsp+E0h] [rbp-1B0h]
  unsigned __int64 v52[16]; // [rsp+F0h] [rbp-1A0h] BYREF
  __m128i v53; // [rsp+170h] [rbp-120h] BYREF
  char v54; // [rsp+188h] [rbp-108h]
  char v55; // [rsp+18Ch] [rbp-104h]
  char v56[64]; // [rsp+190h] [rbp-100h] BYREF
  const __m128i *v57; // [rsp+1D0h] [rbp-C0h]
  const __m128i *v58; // [rsp+1D8h] [rbp-B8h]
  __int8 *v59; // [rsp+1E0h] [rbp-B0h]
  char v60[8]; // [rsp+1E8h] [rbp-A8h] BYREF
  unsigned __int64 v61; // [rsp+1F0h] [rbp-A0h]
  char v62; // [rsp+204h] [rbp-8Ch]
  char v63[64]; // [rsp+208h] [rbp-88h] BYREF
  const __m128i *v64; // [rsp+248h] [rbp-48h]
  unsigned __int64 v65; // [rsp+250h] [rbp-40h]
  unsigned __int64 v66; // [rsp+258h] [rbp-38h]

  sub_11D2BF0((__int64)v42, 0);
  v2 = (_QWORD *)a1[5];
  memset(v52, 0, 0x78u);
  v52[1] = (unsigned __int64)&v52[4];
  BYTE4(v52[3]) = 1;
  LODWORD(v52[2]) = 8;
  v36 = (__int64 *)v2[4];
  v38 = (__int64 *)(*v2 & 0xFFFFFFFFFFFFFFF8LL);
  v43 = 0;
  v44 = v48;
  v45 = 8;
  v46 = 0;
  v47 = 1;
  v49 = 0;
  v50 = 0;
  v51 = 0;
  sub_D695C0((__int64)&v53, (__int64)&v43, v38, (__int64)&v43, (__int64)v36, v3);
  v54 = 0;
  v53.m128i_i64[0] = (__int64)v38;
  sub_298D040((__int64)&v49, &v53);
  sub_D695C0((__int64)&v53, (__int64)&v43, v36, v4, (__int64)v36, v5);
  sub_C8CF70((__int64)&v53, v56, 8, (__int64)v48, (__int64)&v43);
  v57 = v49;
  v6 = v50;
  v50 = 0;
  v58 = (const __m128i *)v6;
  v49 = 0;
  v59 = v51;
  v51 = 0;
  sub_C8CF70((__int64)v60, v63, 8, (__int64)&v52[4], (__int64)v52);
  v10 = v52[12];
  memset(&v52[12], 0, 24);
  v64 = (const __m128i *)v10;
  v65 = v52[13];
  v66 = v52[14];
  if ( v49 )
    j_j___libc_free_0((unsigned __int64)v49);
  if ( !v47 )
    _libc_free((unsigned __int64)v44);
  if ( v52[12] )
    j_j___libc_free_0(v52[12]);
  if ( !BYTE4(v52[3]) )
    _libc_free(v52[1]);
  v11 = (unsigned __int64 *)&v43;
  v12 = (unsigned __int64 *)v48;
  sub_C8CD80((__int64)&v43, (__int64)v48, (__int64)&v53, v7, v8, v9);
  v15 = v58;
  v16 = v57;
  v49 = 0;
  v50 = 0;
  v51 = 0;
  v17 = (char *)v58 - (char *)v57;
  if ( v58 == v57 )
  {
    v18 = 0;
  }
  else
  {
    if ( v17 > 0x7FFFFFFFFFFFFFE0LL )
      goto LABEL_71;
    v18 = (__m128i *)sub_22077B0((char *)v58 - (char *)v57);
    v15 = v58;
    v16 = v57;
  }
  v49 = v18;
  v50 = (__int64)v18;
  v51 = &v18->m128i_i8[v17];
  if ( v16 == v15 )
  {
    v19 = (__int64)v18;
  }
  else
  {
    v19 = (__int64)v18->m128i_i64 + (char *)v15 - (char *)v16;
    do
    {
      if ( v18 )
      {
        *v18 = _mm_loadu_si128(v16);
        v18[1] = _mm_loadu_si128(v16 + 1);
      }
      v18 += 2;
      v16 += 2;
    }
    while ( (__m128i *)v19 != v18 );
  }
  v11 = v52;
  v12 = &v52[4];
  v50 = v19;
  sub_C8CD80((__int64)v52, (__int64)&v52[4], (__int64)v60, v19, v13, v14);
  v20 = (const __m128i *)v65;
  v21 = v64;
  memset(&v52[12], 0, 24);
  v22 = v65 - (_QWORD)v64;
  if ( (const __m128i *)v65 == v64 )
  {
    v24 = 0;
    goto LABEL_20;
  }
  if ( v22 > 0x7FFFFFFFFFFFFFE0LL )
LABEL_71:
    sub_4261EA(v11, v12, v16);
  v23 = sub_22077B0(v65 - (_QWORD)v64);
  v21 = v64;
  v24 = v23;
  v20 = (const __m128i *)v65;
LABEL_20:
  v52[12] = v24;
  v52[13] = v24;
  v52[14] = v24 + v22;
  if ( v21 == v20 )
  {
    v26 = v24;
  }
  else
  {
    v25 = (__m128i *)v24;
    v26 = v24 + (char *)v20 - (char *)v21;
    do
    {
      if ( v25 )
      {
        *v25 = _mm_loadu_si128(v21);
        v25[1] = _mm_loadu_si128(v21 + 1);
      }
      v25 += 2;
      v21 += 2;
    }
    while ( v25 != (__m128i *)v26 );
  }
  for ( v52[13] = v26; ; v26 = v52[13] )
  {
    v27 = (unsigned __int64)v49;
    if ( v50 - (_QWORD)v49 != v26 - v24 )
      goto LABEL_27;
    if ( v49 == (__m128i *)v50 )
      break;
    v33 = v24;
    while ( *(_QWORD *)v27 == *(_QWORD *)v33 )
    {
      v34 = *(_BYTE *)(v27 + 24);
      if ( v34 != *(_BYTE *)(v33 + 24) || v34 && *(_DWORD *)(v27 + 16) != *(_DWORD *)(v33 + 16) )
        break;
      v27 += 32LL;
      v33 += 32LL;
      if ( v50 == v27 )
        goto LABEL_50;
    }
LABEL_27:
    v28 = *(_QWORD *)(v50 - 32);
    for ( i = *(_QWORD *)(v28 + 56); v28 + 48 != i; i = *(_QWORD *)(i + 8) )
    {
      if ( !i )
        BUG();
      v29 = *(_QWORD *)(i - 8);
      v40 = 0;
      v39 = i - 24;
      while ( v29 )
      {
        v30 = v29;
        v29 = *(_QWORD *)(v29 + 8);
        v31 = *(_QWORD *)(v30 + 24);
        if ( v28 != *(_QWORD *)(v31 + 40)
          && (*(_BYTE *)v31 != 84
           || v28 != *(_QWORD *)(*(_QWORD *)(v31 - 8)
                               + 32LL * *(unsigned int *)(v31 + 72)
                               + 8LL * (unsigned int)((v30 - *(_QWORD *)(v31 - 8)) >> 5)))
          && !(unsigned __int8)sub_B19DB0(a1[7], v39, v31) )
        {
          if ( !v40 )
          {
            v41 = sub_ACADE0(*(__int64 ***)(i - 16));
            sub_11D2C80(v42, *(_QWORD *)(i - 16), (unsigned __int8 *)byte_3F871B3, 0);
            v32 = *(_QWORD *)(a1[4] + 80LL);
            if ( v32 )
              v32 -= 24;
            sub_11D33F0(v42, v32, v41);
            sub_11D33F0(v42, v28, v39);
          }
          sub_11D9830(v42, v30);
          v40 = 1;
        }
      }
    }
    sub_23EC7E0((__int64)&v43);
    v24 = v52[12];
  }
LABEL_50:
  if ( v24 )
    j_j___libc_free_0(v24);
  if ( !BYTE4(v52[3]) )
    _libc_free(v52[1]);
  if ( v49 )
    j_j___libc_free_0((unsigned __int64)v49);
  if ( !v47 )
    _libc_free((unsigned __int64)v44);
  if ( v64 )
    j_j___libc_free_0((unsigned __int64)v64);
  if ( !v62 )
    _libc_free(v61);
  if ( v57 )
    j_j___libc_free_0((unsigned __int64)v57);
  if ( !v55 )
    _libc_free(v53.m128i_u64[1]);
  return sub_11D2C20(v42);
}
