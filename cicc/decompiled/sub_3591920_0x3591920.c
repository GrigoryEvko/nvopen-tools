// Function: sub_3591920
// Address: 0x3591920
//
_QWORD *__fastcall sub_3591920(_QWORD *a1, __int64 *a2, __int64 *a3)
{
  __int64 v3; // r13
  __int64 v4; // rax
  const void *v5; // r14
  size_t v6; // r15
  unsigned int v7; // ebx
  int v8; // eax
  unsigned int v9; // r8d
  __int64 i; // rax
  unsigned int v11; // r14d
  __int64 v12; // r15
  unsigned __int64 v13; // rdx
  unsigned __int64 v14; // rsi
  unsigned __int64 v15; // rcx
  int v16; // edi
  char *v17; // rsi
  __int64 v18; // rdx
  int v19; // ecx
  unsigned __int64 v20; // rdx
  int v21; // eax
  unsigned int v22; // r8d
  __int64 v23; // rax
  __int64 v24; // r9
  __int64 v25; // rdx
  unsigned __int64 v26; // rax
  unsigned __int64 v27; // rdi
  __m128i *v28; // rax
  __int64 v29; // rcx
  __m128i *v30; // rdx
  int v31; // eax
  _QWORD *v32; // r8
  int v33; // r14d
  __int64 v34; // rax
  __int64 v35; // rcx
  __int64 v36; // rdx
  __int64 v37; // rax
  __int64 v38; // rax
  __int64 v39; // rdx
  char v40; // di
  __m128i *v41; // rdi
  unsigned __int64 v42; // r8
  __int64 v43; // r12
  __int64 v44; // rbx
  _QWORD *v45; // rdi
  __int64 v47; // rax
  unsigned int v48; // r8d
  __int64 *v49; // r9
  __int64 v50; // rcx
  unsigned int v51; // eax
  __int64 *v52; // rdx
  __int64 v53; // r14
  unsigned __int64 v54; // rdi
  __int64 v55; // [rsp+0h] [rbp-110h]
  __int64 v57; // [rsp+10h] [rbp-100h]
  __int64 v58; // [rsp+18h] [rbp-F8h]
  __int64 *v59; // [rsp+18h] [rbp-F8h]
  __int64 v60; // [rsp+20h] [rbp-F0h]
  unsigned int v61; // [rsp+20h] [rbp-F0h]
  __int64 v62; // [rsp+20h] [rbp-F0h]
  _QWORD *v64; // [rsp+48h] [rbp-C8h]
  unsigned __int64 v65; // [rsp+60h] [rbp-B0h] BYREF
  __int64 v66; // [rsp+68h] [rbp-A8h]
  __int64 v67; // [rsp+70h] [rbp-A0h]
  __m128i *v68; // [rsp+80h] [rbp-90h]
  __int64 v69; // [rsp+88h] [rbp-88h]
  __m128i v70; // [rsp+90h] [rbp-80h] BYREF
  char *v71; // [rsp+A0h] [rbp-70h] BYREF
  size_t v72; // [rsp+A8h] [rbp-68h]
  _QWORD v73[2]; // [rsp+B0h] [rbp-60h] BYREF
  _BYTE *v74; // [rsp+C0h] [rbp-50h] BYREF
  size_t v75; // [rsp+C8h] [rbp-48h]
  _QWORD v76[8]; // [rsp+D0h] [rbp-40h] BYREF

  *((_DWORD *)a1 + 2) = 0;
  a1[2] = 0;
  a1[3] = a1 + 1;
  a1[4] = a1 + 1;
  a1[5] = 0;
  v3 = *a3;
  v67 = 0x1000000000LL;
  v4 = a3[1];
  v65 = 0;
  v66 = 0;
  v64 = a1 + 1;
  v57 = v4;
  if ( v3 == v4 )
  {
    v42 = 0;
    goto LABEL_54;
  }
  do
  {
    v5 = *(const void **)(v3 + 8);
    v6 = *(_QWORD *)(v3 + 16);
    v7 = *(_DWORD *)v3;
    v8 = sub_C92610();
    v9 = sub_C92740((__int64)&v65, v5, v6, v8);
    i = *(_QWORD *)(v65 + 8LL * v9);
    if ( i )
    {
      if ( i != -8 )
        goto LABEL_4;
      LODWORD(v67) = v67 - 1;
    }
    v59 = (__int64 *)(v65 + 8LL * v9);
    v61 = v9;
    v47 = sub_C7D670(v6 + 17, 8);
    v48 = v61;
    v49 = v59;
    v50 = v47;
    if ( v6 )
    {
      v55 = v47;
      memcpy((void *)(v47 + 16), v5, v6);
      v48 = v61;
      v49 = v59;
      v50 = v55;
    }
    *(_BYTE *)(v50 + v6 + 16) = 0;
    *(_QWORD *)v50 = v6;
    *(_DWORD *)(v50 + 8) = 0;
    *v49 = v50;
    ++HIDWORD(v66);
    v51 = sub_C929D0((__int64 *)&v65, v48);
    v52 = (__int64 *)(v65 + 8LL * v51);
    for ( i = *v52; !i; ++v52 )
LABEL_59:
      i = v52[1];
    if ( i == -8 )
      goto LABEL_59;
LABEL_4:
    v11 = *(_DWORD *)(i + 8) + 1;
    *(_DWORD *)(i + 8) = v11;
    if ( v11 <= 9 )
    {
      v71 = (char *)v73;
      sub_2240A50((__int64 *)&v71, 1u, 0);
      v17 = v71;
LABEL_18:
      *v17 = v11 + 48;
      goto LABEL_19;
    }
    if ( v11 <= 0x63 )
    {
      v71 = (char *)v73;
      sub_2240A50((__int64 *)&v71, 2u, 0);
      v17 = v71;
    }
    else
    {
      if ( v11 <= 0x3E7 )
      {
        v14 = 3;
        v12 = v11;
      }
      else
      {
        v12 = v11;
        v13 = v11;
        if ( v11 <= 0x270F )
        {
          v14 = 4;
        }
        else
        {
          LODWORD(v14) = 1;
          while ( 1 )
          {
            v15 = v13;
            v16 = v14;
            v14 = (unsigned int)(v14 + 4);
            v13 /= 0x2710u;
            if ( v15 <= 0x1869F )
              break;
            if ( (unsigned int)v13 <= 0x63 )
            {
              v14 = (unsigned int)(v16 + 5);
              v71 = (char *)v73;
              goto LABEL_14;
            }
            if ( (unsigned int)v13 <= 0x3E7 )
            {
              v14 = (unsigned int)(v16 + 6);
              break;
            }
            if ( (unsigned int)v13 <= 0x270F )
            {
              v14 = (unsigned int)(v16 + 7);
              break;
            }
          }
        }
      }
      v71 = (char *)v73;
LABEL_14:
      sub_2240A50((__int64 *)&v71, v14, 0);
      v17 = v71;
      v18 = v12;
      v19 = v72 - 1;
      while ( 1 )
      {
        v20 = (unsigned __int64)(1374389535 * v18) >> 37;
        v21 = v11 - 100 * v20;
        v22 = v11;
        v11 = v20;
        v23 = (unsigned int)(2 * v21);
        v24 = (unsigned int)(v23 + 1);
        LOBYTE(v23) = a00010203040506[v23];
        v17[v19] = a00010203040506[v24];
        v25 = (unsigned int)(v19 - 1);
        v19 -= 2;
        v17[v25] = v23;
        if ( v22 <= 0x270F )
          break;
        v18 = v11;
      }
      if ( v22 <= 0x3E7 )
        goto LABEL_18;
    }
    v53 = 2 * v11;
    v17[1] = a00010203040506[(unsigned int)(v53 + 1)];
    *v17 = a00010203040506[v53];
LABEL_19:
    v74 = v76;
    sub_35907E0((__int64 *)&v74, *(_BYTE **)(v3 + 8), *(_QWORD *)(v3 + 8) + *(_QWORD *)(v3 + 16));
    if ( v75 == 0x3FFFFFFFFFFFFFFFLL || v75 == 4611686018427387902LL )
      sub_4262D8((__int64)"basic_string::append");
    sub_2241490((unsigned __int64 *)&v74, "__", 2u);
    v26 = 15;
    v27 = 15;
    if ( v74 != (_BYTE *)v76 )
      v27 = v76[0];
    if ( v75 + v72 <= v27 )
      goto LABEL_26;
    if ( v71 != (char *)v73 )
      v26 = v73[0];
    if ( v75 + v72 <= v26 )
    {
      v28 = (__m128i *)sub_2241130((unsigned __int64 *)&v71, 0, 0, v74, v75);
      v68 = &v70;
      v29 = v28->m128i_i64[0];
      v30 = v28 + 1;
      if ( (__m128i *)v28->m128i_i64[0] == &v28[1] )
      {
LABEL_68:
        v70 = _mm_loadu_si128(v28 + 1);
        goto LABEL_28;
      }
    }
    else
    {
LABEL_26:
      v28 = (__m128i *)sub_2241490((unsigned __int64 *)&v74, v71, v72);
      v68 = &v70;
      v29 = v28->m128i_i64[0];
      v30 = v28 + 1;
      if ( (__m128i *)v28->m128i_i64[0] == &v28[1] )
        goto LABEL_68;
    }
    v68 = (__m128i *)v29;
    v70.m128i_i64[0] = v28[1].m128i_i64[0];
LABEL_28:
    v69 = v28->m128i_i64[1];
    v28->m128i_i64[0] = (__int64)v30;
    v28->m128i_i64[1] = 0;
    v28[1].m128i_i8[0] = 0;
    if ( v74 != (_BYTE *)v76 )
      j_j___libc_free_0((unsigned __int64)v74);
    if ( v71 != (char *)v73 )
      j_j___libc_free_0((unsigned __int64)v71);
    v31 = sub_3590AF0(a2, v7, (__int64)v68, v69);
    v32 = v64;
    v33 = v31;
    v34 = a1[2];
    if ( !v34 )
      goto LABEL_39;
    do
    {
      while ( 1 )
      {
        v35 = *(_QWORD *)(v34 + 16);
        v36 = *(_QWORD *)(v34 + 24);
        if ( v7 <= *(_DWORD *)(v34 + 32) )
          break;
        v34 = *(_QWORD *)(v34 + 24);
        if ( !v36 )
          goto LABEL_37;
      }
      v32 = (_QWORD *)v34;
      v34 = *(_QWORD *)(v34 + 16);
    }
    while ( v35 );
LABEL_37:
    if ( v64 == v32 || v7 < *((_DWORD *)v32 + 8) )
    {
LABEL_39:
      v58 = (__int64)v32;
      v37 = sub_22077B0(0x28u);
      *(_DWORD *)(v37 + 32) = v7;
      *(_DWORD *)(v37 + 36) = 0;
      v60 = v37;
      v38 = sub_3591820(a1, v58, (unsigned int *)(v37 + 32));
      if ( v39 )
      {
        v40 = v64 == (_QWORD *)v39 || v38 || v7 < *(_DWORD *)(v39 + 32);
        sub_220F040(v40, v60, (_QWORD *)v39, v64);
        v32 = (_QWORD *)v60;
        ++a1[5];
      }
      else
      {
        v54 = v60;
        v62 = v38;
        j_j___libc_free_0(v54);
        v32 = (_QWORD *)v62;
      }
    }
    v41 = v68;
    *((_DWORD *)v32 + 9) = v33;
    if ( v41 != &v70 )
      j_j___libc_free_0((unsigned __int64)v41);
    v3 += 40;
  }
  while ( v57 != v3 );
  v42 = v65;
  if ( HIDWORD(v66) && (_DWORD)v66 )
  {
    v43 = 8LL * (unsigned int)v66;
    v44 = 0;
    do
    {
      v45 = *(_QWORD **)(v42 + v44);
      if ( v45 != (_QWORD *)-8LL && v45 )
      {
        sub_C7D6A0((__int64)v45, *v45 + 17LL, 8);
        v42 = v65;
      }
      v44 += 8;
    }
    while ( v43 != v44 );
  }
LABEL_54:
  _libc_free(v42);
  return a1;
}
