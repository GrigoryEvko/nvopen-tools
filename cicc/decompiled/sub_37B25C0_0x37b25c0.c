// Function: sub_37B25C0
// Address: 0x37b25c0
//
__int64 __fastcall sub_37B25C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __m128i *v7; // r12
  __m128i *v8; // r13
  __int64 v9; // rcx
  __int64 v10; // rcx
  __int64 i; // rdx
  int v13; // eax
  unsigned int *v14; // rcx
  unsigned int *v15; // r13
  __int64 v16; // rax
  unsigned int v17; // eax
  unsigned int v18; // r14d
  __int64 *v19; // r12
  void *v20; // r15
  char *v21; // rsi
  __int64 v22; // rax
  void *v23; // rdi
  size_t v24; // rdx
  __m128i *v25; // r8
  size_t v26; // rdx
  unsigned int v27; // esi
  _QWORD *v28; // rax
  _QWORD *v29; // rax
  __int64 v30; // rsi
  unsigned int v31; // eax
  int v32; // eax
  unsigned __int64 v33; // rax
  __int64 v34; // rax
  int v35; // r15d
  __int64 v36; // r13
  __int64 v37; // rax
  __int64 v38; // rdi
  __int64 v39; // r8
  unsigned int v40; // eax
  int v41; // eax
  unsigned __int64 v42; // rax
  __int64 v43; // rax
  int v44; // ecx
  __int64 v45; // r13
  __int64 v46; // rax
  _QWORD *v47; // rsi
  __int64 v48; // [rsp+8h] [rbp-E8h]
  int v49; // [rsp+20h] [rbp-D0h]
  __int64 v50; // [rsp+30h] [rbp-C0h]
  int v51; // [rsp+30h] [rbp-C0h]
  int v53; // [rsp+38h] [rbp-B8h]
  void *p_src; // [rsp+40h] [rbp-B0h] BYREF
  __int64 v55; // [rsp+48h] [rbp-A8h]
  void *src; // [rsp+50h] [rbp-A0h] BYREF
  __int64 v57; // [rsp+58h] [rbp-98h]
  __int64 v58; // [rsp+60h] [rbp-90h] BYREF
  _QWORD *v59; // [rsp+68h] [rbp-88h]
  __int64 v60; // [rsp+70h] [rbp-80h]
  __int64 v61; // [rsp+78h] [rbp-78h]
  char *v62; // [rsp+80h] [rbp-70h] BYREF
  __int64 v63; // [rsp+88h] [rbp-68h]
  __int64 v64; // [rsp+90h] [rbp-60h] BYREF
  _QWORD *v65; // [rsp+98h] [rbp-58h]
  __int64 v66; // [rsp+A0h] [rbp-50h]
  __int64 v67; // [rsp+A8h] [rbp-48h]
  char *v68; // [rsp+B0h] [rbp-40h] BYREF
  __int64 v69; // [rsp+B8h] [rbp-38h]
  _BYTE v70[48]; // [rsp+C0h] [rbp-30h] BYREF

  v62 = (char *)&v64;
  v7 = *(__m128i **)a1;
  v58 = 0;
  v8 = v7 + 4;
  v59 = 0;
  v60 = 0;
  v61 = 0;
  v63 = 0;
  v64 = 0;
  v65 = 0;
  v66 = 0;
  v67 = 0;
  v68 = v70;
  v69 = 0;
  do
  {
    LODWORD(v57) = 1;
    src = 0;
    v9 = v7->m128i_u32[2];
    if ( (_DWORD)v9 )
      goto LABEL_3;
    if ( !sub_33D1410(v7->m128i_i64[0], (__int64)&src, a3, v9, a5) && !(unsigned __int8)sub_33CA6D0(v7->m128i_i64[0]) )
    {
      if ( (unsigned int)v57 > 0x40 && src )
        j_j___libc_free_0_0((unsigned __int64)src);
LABEL_3:
      if ( *(_DWORD *)(v7->m128i_i64[0] + 24) != 51 )
        sub_37B2430((__int64)&v58, v7);
      goto LABEL_5;
    }
    if ( (unsigned int)v57 > 0x40 && src )
      j_j___libc_free_0_0((unsigned __int64)src);
    sub_37B2430((__int64)&v64, v7);
LABEL_5:
    ++v7;
  }
  while ( v8 != v7 );
  v10 = (unsigned int)v63;
  if ( (_DWORD)v63 == 4 )
    goto LABEL_7;
  ++v58;
  if ( (_DWORD)v60 )
  {
    v27 = 4 * v60;
    i = (unsigned int)v61;
    if ( (unsigned int)(4 * v60) < 0x40 )
      v27 = 64;
    if ( (unsigned int)v61 <= v27 )
      goto LABEL_55;
    a5 = (__int64)v59;
    v30 = 2LL * (unsigned int)v61;
    if ( (_DWORD)v60 == 1 )
    {
      v36 = 2048;
      v35 = 128;
    }
    else
    {
      _BitScanReverse(&v31, v60 - 1);
      v32 = 1 << (33 - (v31 ^ 0x1F));
      if ( v32 < 64 )
        v32 = 64;
      if ( v32 == (_DWORD)v61 )
      {
        v60 = 0;
        v47 = &v59[v30];
        do
        {
          if ( a5 )
          {
            *(_QWORD *)a5 = 0;
            *(_DWORD *)(a5 + 8) = -1;
          }
          a5 += 16;
        }
        while ( v47 != (_QWORD *)a5 );
        goto LABEL_21;
      }
      v33 = (4 * v32 / 3u + 1) | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1);
      v34 = ((((v33 >> 2) | v33 | (((v33 >> 2) | v33) >> 4)) >> 8)
           | (v33 >> 2)
           | v33
           | (((v33 >> 2) | v33) >> 4)
           | (((((v33 >> 2) | v33 | (((v33 >> 2) | v33) >> 4)) >> 8) | (v33 >> 2) | v33 | (((v33 >> 2) | v33) >> 4)) >> 16))
          + 1;
      v35 = v34;
      v36 = 16 * v34;
    }
    sub_C7D6A0((__int64)v59, v30 * 8, 8);
    LODWORD(v61) = v35;
    v37 = sub_C7D670(v36, 8);
    v60 = 0;
    v59 = (_QWORD *)v37;
    for ( i = v37 + 16LL * (unsigned int)v61; i != v37; v37 += 16 )
    {
      if ( v37 )
      {
        *(_QWORD *)v37 = 0;
        *(_DWORD *)(v37 + 8) = -1;
      }
    }
LABEL_21:
    v10 = (unsigned int)v63;
    goto LABEL_22;
  }
  i = HIDWORD(v60);
  if ( HIDWORD(v60) )
  {
    i = (unsigned int)v61;
    if ( (unsigned int)v61 > 0x40 )
    {
      sub_C7D6A0((__int64)v59, 16LL * (unsigned int)v61, 8);
      v60 = 0;
      v59 = 0;
      LODWORD(v61) = 0;
      goto LABEL_21;
    }
LABEL_55:
    v28 = v59;
    i = (__int64)&v59[2 * i];
    if ( v59 != (_QWORD *)i )
    {
      do
      {
        *v28 = 0;
        v28 += 2;
        *((_DWORD *)v28 - 2) = -1;
      }
      while ( (_QWORD *)i != v28 );
      v10 = (unsigned int)v63;
    }
    v60 = 0;
  }
LABEL_22:
  v55 = 0;
  p_src = &src;
  if ( (_DWORD)v10 )
    sub_3774680((__int64)&p_src, &v62, i, v10, a5, a6);
  ++v64;
  if ( (_DWORD)v66 )
  {
    v10 = (unsigned int)(4 * v66);
    i = (unsigned int)v67;
    if ( (unsigned int)v10 < 0x40 )
      v10 = 64;
    if ( (unsigned int)v67 <= (unsigned int)v10 )
      goto LABEL_62;
    v38 = (__int64)v65;
    v39 = 2LL * (unsigned int)v67;
    if ( (_DWORD)v66 == 1 )
    {
      v45 = 2048;
      v44 = 128;
    }
    else
    {
      _BitScanReverse(&v40, v66 - 1);
      v10 = 33 - (v40 ^ 0x1F);
      v41 = 1 << (33 - (v40 ^ 0x1F));
      if ( v41 < 64 )
        v41 = 64;
      if ( (_DWORD)v67 == v41 )
      {
        v66 = 0;
        a5 = (__int64)&v65[v39];
        do
        {
          if ( v38 )
          {
            *(_QWORD *)v38 = 0;
            *(_DWORD *)(v38 + 8) = -1;
          }
          v38 += 16;
        }
        while ( a5 != v38 );
        goto LABEL_28;
      }
      v42 = (4 * v41 / 3u + 1) | ((unsigned __int64)(4 * v41 / 3u + 1) >> 1);
      v43 = ((((v42 >> 2) | v42 | (((v42 >> 2) | v42) >> 4)) >> 8)
           | (v42 >> 2)
           | v42
           | (((v42 >> 2) | v42) >> 4)
           | (((((v42 >> 2) | v42 | (((v42 >> 2) | v42) >> 4)) >> 8) | (v42 >> 2) | v42 | (((v42 >> 2) | v42) >> 4)) >> 16))
          + 1;
      v44 = v43;
      v45 = 16 * v43;
    }
    v51 = v44;
    sub_C7D6A0((__int64)v65, v39 * 8, 8);
    LODWORD(v67) = v51;
    v46 = sub_C7D670(v45, 8);
    v66 = 0;
    v65 = (_QWORD *)v46;
    for ( i = v46 + 16LL * (unsigned int)v67; i != v46; v46 += 16 )
    {
      if ( v46 )
      {
        *(_QWORD *)v46 = 0;
        *(_DWORD *)(v46 + 8) = -1;
      }
    }
    goto LABEL_28;
  }
  if ( !HIDWORD(v66) )
    goto LABEL_28;
  i = (unsigned int)v67;
  if ( (unsigned int)v67 > 0x40 )
  {
    sub_C7D6A0((__int64)v65, 16LL * (unsigned int)v67, 8);
    v65 = 0;
    v66 = 0;
    LODWORD(v67) = 0;
    goto LABEL_28;
  }
LABEL_62:
  v29 = v65;
  i = (__int64)&v65[2 * i];
  if ( v65 != (_QWORD *)i )
  {
    do
    {
      *v29 = 0;
      v29 += 2;
      *((_DWORD *)v29 - 2) = -1;
    }
    while ( (_QWORD *)i != v29 );
  }
  v66 = 0;
LABEL_28:
  v57 = 0;
  src = &v58;
  v13 = v69;
  if ( (_DWORD)v69 )
  {
    sub_3774680((__int64)&src, &v68, i, v10, a5, a6);
    v13 = v57;
    v14 = *(unsigned int **)a2;
    v50 = *(_QWORD *)a2 + 4LL * *(unsigned int *)(a2 + 8);
    if ( *(_QWORD *)a2 == v50 )
    {
      v48 = 16LL * (unsigned int)v57;
      v24 = v48;
LABEL_40:
      v25 = (__m128i *)(*(_QWORD *)a1 + v48);
      if ( v24 )
      {
        memmove(*(void **)a1, src, v24);
        v25 = (__m128i *)(v48 + *(_QWORD *)a1);
      }
      v26 = 16LL * (unsigned int)v55;
      if ( !v26 )
        goto LABEL_44;
      goto LABEL_43;
    }
    v48 = 16LL * (unsigned int)v57;
LABEL_30:
    v49 = v13;
    v15 = v14;
    while ( 1 )
    {
      while ( 1 )
      {
        v17 = *v15;
        if ( *v15 != -1 )
          break;
LABEL_33:
        if ( ++v15 == (unsigned int *)v50 )
          goto LABEL_39;
      }
      v18 = *(_DWORD *)(a1 + 16);
      v53 = v17 % v18;
      v19 = (__int64 *)(*(_QWORD *)a1 + 16LL * (v17 / v18));
      if ( *(_DWORD *)(*v19 + 24) != 51 )
      {
        v20 = src;
        v21 = (char *)src + 16 * (unsigned int)v57;
        v22 = sub_37747E0((__int64)src, (__int64)v21, v19);
        if ( v21 == (char *)v22 )
        {
          v23 = p_src;
          LODWORD(v16) = v49
                       + ((sub_37747E0((__int64)p_src, (__int64)p_src + 16 * (unsigned int)v55, v19) - (__int64)v23) >> 4);
        }
        else
        {
          v16 = (v22 - (__int64)v20) >> 4;
        }
        *v15 = v16 * v18 + v53;
        goto LABEL_33;
      }
      *v15++ = -1;
      if ( v15 == (unsigned int *)v50 )
      {
LABEL_39:
        v24 = 16LL * (unsigned int)v57;
        goto LABEL_40;
      }
    }
  }
  v48 = 0;
  v14 = *(unsigned int **)a2;
  v50 = *(_QWORD *)a2 + 4LL * *(unsigned int *)(a2 + 8);
  if ( v50 != *(_QWORD *)a2 )
    goto LABEL_30;
  v25 = *(__m128i **)a1;
  v26 = 16LL * (unsigned int)v55;
  if ( v26 )
  {
LABEL_43:
    memmove(v25, p_src, v26);
LABEL_44:
    if ( src != &v58 )
      _libc_free((unsigned __int64)src);
  }
  if ( p_src != &src )
    _libc_free((unsigned __int64)p_src);
LABEL_7:
  if ( v68 != v70 )
    _libc_free((unsigned __int64)v68);
  sub_C7D6A0((__int64)v65, 16LL * (unsigned int)v67, 8);
  if ( v62 != (char *)&v64 )
    _libc_free((unsigned __int64)v62);
  return sub_C7D6A0((__int64)v59, 16LL * (unsigned int)v61, 8);
}
