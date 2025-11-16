// Function: sub_31CCB00
// Address: 0x31ccb00
//
__int64 __fastcall sub_31CCB00(__int64 a1, __int64 a2, __int64 a3)
{
  int *v3; // rbx
  __int64 v4; // r14
  __int64 v5; // rax
  __int64 v6; // r15
  __int64 v7; // rdx
  __int64 v8; // r8
  __int64 v9; // r9
  unsigned __int64 v10; // rdi
  const __m128i *v11; // rbx
  const __m128i *v12; // r13
  __m128i *v13; // rsi
  unsigned __int64 v14; // rcx
  __m128i *v15; // rdx
  const __m128i *v16; // rax
  const __m128i *v17; // rsi
  const __m128i *v18; // rsi
  unsigned __int64 v19; // rdi
  const __m128i *v20; // rbx
  const __m128i *v21; // r13
  __m128i *v22; // rsi
  __m128i *v23; // rdx
  const __m128i *v24; // rax
  const __m128i *v25; // rcx
  __int64 v26; // rdi
  __int64 v27; // rsi
  _QWORD **v28; // r12
  _QWORD **v30; // rbx
  int *v31; // [rsp+10h] [rbp-7A0h]
  unsigned __int8 v32; // [rsp+1Fh] [rbp-791h]
  __int64 v33; // [rsp+28h] [rbp-788h]
  int *v34; // [rsp+28h] [rbp-788h]
  __int64 v38; // [rsp+58h] [rbp-758h]
  unsigned __int64 v39; // [rsp+60h] [rbp-750h] BYREF
  __m128i *v40; // [rsp+68h] [rbp-748h]
  const __m128i *v41; // [rsp+70h] [rbp-740h]
  unsigned __int64 v42; // [rsp+80h] [rbp-730h] BYREF
  __m128i *v43; // [rsp+88h] [rbp-728h]
  const __m128i *v44; // [rsp+90h] [rbp-720h]
  __int64 v45; // [rsp+A0h] [rbp-710h] BYREF
  _QWORD **v46; // [rsp+A8h] [rbp-708h]
  __int64 v47; // [rsp+B0h] [rbp-700h]
  __int64 v48; // [rsp+B8h] [rbp-6F8h]
  int v49[4]; // [rsp+C0h] [rbp-6F0h] BYREF
  __int64 v50; // [rsp+D0h] [rbp-6E0h]
  unsigned int v51; // [rsp+E0h] [rbp-6D0h]
  __int64 v52; // [rsp+F0h] [rbp-6C0h] BYREF
  __int64 v53; // [rsp+F8h] [rbp-6B8h]
  int *v54; // [rsp+100h] [rbp-6B0h]
  __int64 v55; // [rsp+108h] [rbp-6A8h]
  __int64 v56; // [rsp+110h] [rbp-6A0h]
  __int64 v57; // [rsp+118h] [rbp-698h]
  __int64 v58; // [rsp+120h] [rbp-690h]
  __int64 v59[7]; // [rsp+128h] [rbp-688h] BYREF
  const __m128i *v60; // [rsp+160h] [rbp-650h]
  __m128i *v61; // [rsp+168h] [rbp-648h]
  const __m128i *v62; // [rsp+170h] [rbp-640h]
  const __m128i *v63; // [rsp+178h] [rbp-638h]
  __m128i *v64; // [rsp+180h] [rbp-630h]
  const __m128i *v65; // [rsp+188h] [rbp-628h]
  _BYTE *v66; // [rsp+190h] [rbp-620h]
  __int64 v67; // [rsp+198h] [rbp-618h]
  _BYTE v68[128]; // [rsp+1A0h] [rbp-610h] BYREF
  __int64 v69; // [rsp+220h] [rbp-590h]
  __int64 v70; // [rsp+228h] [rbp-588h]
  __int64 v71; // [rsp+230h] [rbp-580h]
  __int64 v72; // [rsp+238h] [rbp-578h]
  __int64 v73; // [rsp+240h] [rbp-570h]
  __int64 v74; // [rsp+248h] [rbp-568h]
  __int64 v75[2]; // [rsp+250h] [rbp-560h] BYREF
  unsigned __int64 v76; // [rsp+260h] [rbp-550h]
  _BYTE *v77; // [rsp+268h] [rbp-548h]
  __int64 v78; // [rsp+270h] [rbp-540h]
  _BYTE v79[1280]; // [rsp+278h] [rbp-538h] BYREF
  __int64 *v80; // [rsp+778h] [rbp-38h]

  v3 = v49;
  v4 = *(_QWORD *)(**(_QWORD **)(a1 + 96) + 72LL);
  sub_31CA2C0(v49, a1);
  v5 = *(_QWORD *)(v4 + 80);
  v45 = 0;
  v46 = 0;
  v47 = 0;
  v48 = 0;
  if ( !v5 )
    BUG();
  v6 = *(_QWORD *)(v5 + 32);
  v38 = v5 + 24;
  if ( v6 == v5 + 24 )
  {
    v32 = 0;
    v27 = 0;
    v26 = 0;
    goto LABEL_45;
  }
  v32 = 0;
  do
  {
    if ( !v6 )
      BUG();
    if ( *(_BYTE *)(v6 - 24) == 60 )
    {
      v75[0] = v6 - 24;
      v75[1] = a1;
      v76 = 0;
      v77 = v79;
      v78 = 0x2000000000LL;
      v80 = &v45;
      if ( !sub_31CB6C0(v75, (__int64)v3) )
      {
        sub_2A4DA70(v6 - 24, 1);
        sub_31CC260(v75);
        sub_31CADB0(v75);
        v32 = 1;
LABEL_11:
        if ( v77 != v79 )
          _libc_free((unsigned __int64)v77);
        if ( v76 )
          j_j___libc_free_0(v76);
        goto LABEL_15;
      }
      v54 = v3;
      v7 = *(_QWORD *)(v4 + 40);
      v52 = a1;
      v57 = v6 - 24;
      v58 = 0;
      v55 = v7 + 312;
      v53 = a2;
      v56 = a3;
      v66 = v68;
      v67 = 0x1000000000LL;
      v59[0] = 0;
      v59[1] = -1;
      memset(&v59[2], 0, 40);
      v60 = 0;
      v61 = 0;
      v62 = 0;
      v63 = 0;
      v64 = 0;
      v65 = 0;
      v69 = 0;
      v70 = 0;
      v71 = 0;
      v72 = 0;
      v73 = 0;
      v74 = 0;
      v33 = sub_31CC7B0((__int64)&v52, (__int64)v3, v7, a2, v8, v9);
      if ( !v33 && v58 )
      {
        v39 = 0;
        v40 = 0;
        v41 = 0;
        v10 = (unsigned __int64)v61;
        if ( v60 != v61 )
        {
          v31 = v3;
          v11 = v60;
          v12 = v61;
          while ( 1 )
          {
            while ( !(unsigned __int8)sub_31C7EB0(&v11->m128i_i64[1], v59, v53) )
            {
LABEL_23:
              v11 = (const __m128i *)((char *)v11 + 72);
              if ( v12 == v11 )
                goto LABEL_29;
            }
            v13 = v40;
            if ( v40 == v41 )
            {
              sub_31C9090(&v39, v40, v11);
              goto LABEL_23;
            }
            if ( v40 )
            {
              *v40 = _mm_loadu_si128(v11);
              v13[1] = _mm_loadu_si128(v11 + 1);
              v13[2] = _mm_loadu_si128(v11 + 2);
              v13[3] = _mm_loadu_si128(v11 + 3);
              v13[4].m128i_i64[0] = v11[4].m128i_i64[0];
              v13 = v40;
            }
            v11 = (const __m128i *)((char *)v11 + 72);
            v40 = (__m128i *)((char *)v13 + 72);
            if ( v12 == v11 )
            {
LABEL_29:
              v3 = v31;
              v14 = v39;
              v15 = v40;
              v10 = (unsigned __int64)v60;
              v16 = v41;
              v17 = v61;
              goto LABEL_30;
            }
          }
        }
        v17 = v61;
        v15 = 0;
        v16 = 0;
        v14 = 0;
LABEL_30:
        v40 = (__m128i *)v17;
        v18 = v62;
        v60 = (const __m128i *)v14;
        v62 = v16;
        v39 = v10;
        v41 = v18;
        v19 = (unsigned __int64)v64;
        v61 = v15;
        v42 = 0;
        v43 = 0;
        v44 = 0;
        if ( v63 != v64 )
        {
          v34 = v3;
          v20 = v63;
          v21 = v64;
          while ( 1 )
          {
            while ( (unsigned __int8)sub_31C7EB0(&v20->m128i_i64[1], v59, v53) != 3 )
            {
LABEL_32:
              v20 = (const __m128i *)((char *)v20 + 72);
              if ( v21 == v20 )
                goto LABEL_38;
            }
            v22 = v43;
            if ( v43 == v44 )
            {
              sub_31C9090(&v42, v43, v20);
              goto LABEL_32;
            }
            if ( v43 )
            {
              *v43 = _mm_loadu_si128(v20);
              v22[1] = _mm_loadu_si128(v20 + 1);
              v22[2] = _mm_loadu_si128(v20 + 2);
              v22[3] = _mm_loadu_si128(v20 + 3);
              v22[4].m128i_i64[0] = v20[4].m128i_i64[0];
              v22 = v43;
            }
            v20 = (const __m128i *)((char *)v20 + 72);
            v43 = (__m128i *)((char *)v22 + 72);
            if ( v21 == v20 )
            {
LABEL_38:
              v3 = v34;
              v23 = v43;
              v33 = v42;
              v19 = (unsigned __int64)v63;
              v24 = v44;
              v25 = v64;
              goto LABEL_39;
            }
          }
        }
        v25 = v64;
        v23 = 0;
        v24 = 0;
LABEL_39:
        v43 = (__m128i *)v25;
        v42 = v19;
        v44 = v65;
        v63 = (const __m128i *)v33;
        v64 = v23;
        v65 = v24;
        if ( v19 )
          j_j___libc_free_0(v19);
        if ( v39 )
          j_j___libc_free_0(v39);
        sub_31C9A20(&v52);
      }
      sub_C7D6A0(v70, 8LL * (unsigned int)v72, 8);
      if ( v66 != v68 )
        _libc_free((unsigned __int64)v66);
      if ( v63 )
        j_j___libc_free_0((unsigned __int64)v63);
      if ( v60 )
        j_j___libc_free_0((unsigned __int64)v60);
      goto LABEL_11;
    }
LABEL_15:
    v6 = *(_QWORD *)(v6 + 8);
  }
  while ( v38 != v6 );
  v26 = (__int64)v46;
  v27 = (unsigned int)v48;
  v28 = &v46[v27];
  if ( (_DWORD)v47 && v46 != v28 )
  {
    v30 = v46;
    while ( *v30 == (_QWORD *)-4096LL || *v30 == (_QWORD *)-8192LL )
    {
      if ( ++v30 == v28 )
        goto LABEL_45;
    }
    if ( v30 != v28 )
    {
LABEL_52:
      sub_B43D60(*v30);
      while ( ++v30 != v28 )
      {
        if ( *v30 != (_QWORD *)-8192LL && *v30 != (_QWORD *)-4096LL )
        {
          if ( v30 != v28 )
            goto LABEL_52;
          break;
        }
      }
      v26 = (__int64)v46;
      v27 = (unsigned int)v48;
    }
  }
LABEL_45:
  sub_C7D6A0(v26, v27 * 8, 8);
  sub_C7D6A0(v50, 16LL * v51, 8);
  return v32;
}
