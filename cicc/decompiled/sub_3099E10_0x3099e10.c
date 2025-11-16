// Function: sub_3099E10
// Address: 0x3099e10
//
unsigned __int64 __fastcall sub_3099E10(__int64 a1, _BYTE **a2, unsigned __int64 *a3, unsigned __int64 *a4, __int64 a5)
{
  __int64 *v5; // rax
  __int64 ***v6; // rbx
  _BYTE *v7; // r14
  char v8; // si
  __int64 **v9; // r15
  __int64 *v10; // rbx
  int v11; // ecx
  _QWORD *v12; // r13
  int v13; // r14d
  __int64 v14; // rsi
  __int64 v15; // rdx
  __int64 v16; // rcx
  char v17; // r12
  __int64 v18; // r12
  int v19; // ecx
  __int64 i; // r14
  unsigned __int64 v21; // rax
  __int64 v22; // r15
  const char *v23; // r12
  size_t v24; // rdx
  size_t v25; // r13
  unsigned __int64 v26; // rbx
  int v27; // eax
  int v28; // eax
  char *v29; // r14
  _QWORD **v30; // rax
  size_t v31; // r13
  _QWORD *v32; // rdx
  unsigned __int64 *v33; // rdi
  __int64 v34; // rcx
  unsigned __int64 v35; // rdx
  unsigned __int64 v36; // rsi
  __int64 *v37; // rax
  __int64 v38; // rsi
  _QWORD *v39; // rbx
  _QWORD *v40; // r12
  __int64 v41; // rsi
  __int64 v42; // r12
  __int64 v43; // rsi
  unsigned __int64 v44; // r8
  __int64 v45; // r12
  __int64 v46; // rbx
  _QWORD *v47; // rdi
  __int64 *v49; // rax
  __int64 v50; // r13
  __int64 v51; // rax
  __int64 *v52; // rbx
  __int64 v53; // r14
  __int64 v54; // rdi
  __int64 v55; // rdx
  __int64 v56; // rcx
  char *v57; // r12
  unsigned __int64 v58; // rax
  __int64 v59; // rdx
  __int64 v60; // rcx
  unsigned __int64 v61; // r12
  _QWORD *v62; // rdi
  __int64 v63; // rdx
  unsigned __int64 v65; // [rsp+8h] [rbp-138h]
  _QWORD **v66; // [rsp+10h] [rbp-130h]
  __int64 *v69; // [rsp+30h] [rbp-110h]
  _QWORD *v70; // [rsp+38h] [rbp-108h]
  int v71; // [rsp+38h] [rbp-108h]
  char *s; // [rsp+48h] [rbp-F8h] BYREF
  _QWORD **v73[2]; // [rsp+50h] [rbp-F0h] BYREF
  unsigned __int64 v74; // [rsp+60h] [rbp-E0h] BYREF
  __int64 v75; // [rsp+68h] [rbp-D8h]
  __int64 v76; // [rsp+70h] [rbp-D0h]
  __m128i v77; // [rsp+80h] [rbp-C0h] BYREF
  _QWORD src[2]; // [rsp+90h] [rbp-B0h] BYREF
  __int64 v79[4]; // [rsp+A0h] [rbp-A0h] BYREF
  unsigned int v80; // [rsp+C0h] [rbp-80h]
  __int64 *v81; // [rsp+D0h] [rbp-70h]
  unsigned int v82; // [rsp+E0h] [rbp-60h]
  _QWORD *v83; // [rsp+F0h] [rbp-50h]
  unsigned int v84; // [rsp+100h] [rbp-40h]

  v5 = *(__int64 **)(a1 + 8);
  v6 = *(__int64 ****)a1;
  v74 = 0;
  v75 = 0;
  v7 = *a2;
  v69 = v5;
  v8 = *(_BYTE *)(a5 + 232);
  v76 = 0x800000000LL;
  v9 = *v6;
  v65 = (unsigned __int64)*v6;
  sub_B6F950(**v6, v8);
  if ( (*v7 & 1) != 0 )
    sub_3099AF0((__int64)v9, (__int64)&v74);
  v10 = (__int64 *)(v6 + 1);
  sub_E48650(v79, v65);
  v11 = 1;
  v12 = v7;
  while ( 1 )
  {
    v71 = v11;
    if ( v69 == v10 )
      break;
    v18 = *v10++;
    sub_2240AE0(a4, (unsigned __int64 *)(v18 + 168));
    sub_B6F950(*(__int64 **)v18, *(_BYTE *)(a5 + 232));
    v19 = v71;
    if ( v71 == 63 )
    {
      v13 = 0;
      v70 = v12 + 1;
    }
    else
    {
      v70 = v12;
      v13 = v19 + 1;
    }
    if ( (*v12 & (1LL << v19)) != 0 )
      sub_3099AF0(v18, (__int64)&v74);
    v14 = (__int64)v73;
    v73[0] = (_QWORD **)v18;
    src[0] = 0;
    v17 = sub_E49E40(v79, (__int64)v73, 0, &v77);
    if ( v73[0] )
    {
      v66 = v73[0];
      sub_BA9C10(v73[0], (__int64)v73, v15, v16);
      v14 = 880;
      j_j___libc_free_0((unsigned __int64)v66);
    }
    if ( src[0] )
    {
      v14 = (__int64)&v77;
      ((void (__fastcall *)(__m128i *, __m128i *, __int64))src[0])(&v77, &v77, 3);
    }
    if ( v17 )
    {
      v77.m128i_i64[0] = 0;
      sub_CEAF80(v77.m128i_i64);
      v57 = (char *)v77.m128i_i64[0];
      if ( v77.m128i_i64[0] )
      {
        v58 = strlen((const char *)v77.m128i_i64[0]);
        if ( v58 > 0x3FFFFFFFFFFFFFFFLL - a3[1] )
          sub_4262D8((__int64)"basic_string::append");
        v14 = (__int64)v57;
        sub_2241490(a3, v57, v58);
        if ( v77.m128i_i64[0] )
          j_j___libc_free_0_0(v77.m128i_u64[0]);
      }
      sub_BA9C10((_QWORD **)v65, v14, v55, v56);
      j_j___libc_free_0(v65);
      for ( ; v69 != v10; ++v10 )
      {
        v61 = *v10;
        if ( *v10 )
        {
          sub_BA9C10((_QWORD **)*v10, 880, v59, v60);
          j_j___libc_free_0(v61);
        }
      }
      v65 = 0;
      goto LABEL_41;
    }
    v12 = v70;
    v11 = v13;
  }
  for ( i = *(_QWORD *)(v65 + 32); v65 + 24 != i; i = *(_QWORD *)(i + 8) )
  {
    v22 = 0;
    if ( i )
      v22 = i - 56;
    v23 = sub_BD5D20(v22);
    v25 = v24;
    v26 = v74 + 8LL * (unsigned int)v75;
    v27 = sub_C92610();
    v28 = sub_C92860((__int64 *)&v74, v23, v25, v27);
    if ( v28 == -1 )
      v21 = v74 + 8LL * (unsigned int)v75;
    else
      v21 = v74 + 8LL * v28;
    if ( v26 != v21 )
      *(_WORD *)(v22 + 32) = *(_WORD *)(v22 + 32) & 0xBCC0 | 0x4007;
  }
  s = 0;
  sub_CEAF80((__int64 *)&s);
  v29 = s;
  if ( s )
  {
    v77.m128i_i64[0] = (__int64)src;
    v30 = (_QWORD **)strlen(s);
    v73[0] = v30;
    v31 = (size_t)v30;
    if ( (unsigned __int64)v30 > 0xF )
    {
      v77.m128i_i64[0] = sub_22409D0((__int64)&v77, (unsigned __int64 *)v73, 0);
      v62 = (_QWORD *)v77.m128i_i64[0];
      src[0] = v73[0];
    }
    else
    {
      if ( v30 == (_QWORD **)1 )
      {
        LOBYTE(src[0]) = *v29;
        v32 = src;
        goto LABEL_30;
      }
      if ( !v30 )
      {
        v32 = src;
LABEL_30:
        v77.m128i_i64[1] = (__int64)v30;
        *((_BYTE *)v30 + (_QWORD)v32) = 0;
        v33 = (unsigned __int64 *)*a3;
        if ( (_QWORD *)v77.m128i_i64[0] == src )
        {
          v63 = v77.m128i_i64[1];
          if ( v77.m128i_i64[1] )
          {
            if ( v77.m128i_i64[1] == 1 )
              *(_BYTE *)v33 = src[0];
            else
              memcpy(v33, src, v77.m128i_u64[1]);
            v63 = v77.m128i_i64[1];
            v33 = (unsigned __int64 *)*a3;
          }
          a3[1] = v63;
          *((_BYTE *)v33 + v63) = 0;
          v33 = (unsigned __int64 *)v77.m128i_i64[0];
          goto LABEL_34;
        }
        v34 = v77.m128i_i64[1];
        v35 = src[0];
        if ( v33 == a3 + 2 )
        {
          *a3 = v77.m128i_i64[0];
          a3[1] = v34;
          a3[2] = v35;
        }
        else
        {
          v36 = a3[2];
          *a3 = v77.m128i_i64[0];
          a3[1] = v34;
          a3[2] = v35;
          if ( v33 )
          {
            v77.m128i_i64[0] = (__int64)v33;
            src[0] = v36;
LABEL_34:
            v77.m128i_i64[1] = 0;
            *(_BYTE *)v33 = 0;
            if ( (_QWORD *)v77.m128i_i64[0] != src )
              j_j___libc_free_0(v77.m128i_u64[0]);
            if ( s )
              j_j___libc_free_0_0((unsigned __int64)s);
            s = 0;
            goto LABEL_39;
          }
        }
        v77.m128i_i64[0] = (__int64)src;
        v33 = src;
        goto LABEL_34;
      }
      v62 = src;
    }
    memcpy(v62, v29, v31);
    v30 = v73[0];
    v32 = (_QWORD *)v77.m128i_i64[0];
    goto LABEL_30;
  }
LABEL_39:
  sub_B848C0(v73);
  v37 = sub_2D028E0((__int64 *)(a5 + 208));
  sub_B8B500((__int64)v73, v37, 1u);
  if ( (unsigned __int8)sub_B89FE0((__int64)v73, v65) && !LOBYTE(qword_502D788[8]) )
  {
    sub_B848C0(&v77);
    v49 = (__int64 *)sub_2D1B100();
    sub_B8B500((__int64)&v77, v49, 0);
    sub_B89FE0((__int64)&v77, v65);
    sub_B82680(&v77);
  }
  sub_B82680(v73);
LABEL_41:
  v38 = v84;
  if ( v84 )
  {
    v39 = v83;
    v40 = &v83[2 * v84];
    do
    {
      if ( *v39 != -4096 && *v39 != -8192 )
      {
        v41 = v39[1];
        if ( v41 )
          sub_B91220((__int64)(v39 + 1), v41);
      }
      v39 += 2;
    }
    while ( v40 != v39 );
    v38 = v84;
  }
  sub_C7D6A0((__int64)v83, 16 * v38, 8);
  if ( v82 )
  {
    v50 = sub_1061AC0();
    v51 = sub_1061AD0();
    v52 = v81;
    v53 = v51;
    v43 = v82;
    v42 = (__int64)&v81[v43];
    if ( v81 != &v81[v43] )
    {
      do
      {
        while ( sub_1061B40(*v52, v50) )
        {
          if ( (__int64 *)v42 == ++v52 )
            goto LABEL_68;
        }
        v54 = *v52++;
        sub_1061B40(v54, v53);
      }
      while ( (__int64 *)v42 != v52 );
LABEL_68:
      v42 = (__int64)v81;
      v43 = v82;
    }
  }
  else
  {
    v42 = (__int64)v81;
    v43 = 0;
  }
  sub_C7D6A0(v42, v43 * 8, 8);
  sub_C7D6A0(v79[2], 8LL * v80, 8);
  if ( HIDWORD(v75) )
  {
    v44 = v74;
    if ( (_DWORD)v75 )
    {
      v45 = 8LL * (unsigned int)v75;
      v46 = 0;
      do
      {
        v47 = *(_QWORD **)(v44 + v46);
        if ( v47 != (_QWORD *)-8LL && v47 )
        {
          sub_C7D6A0((__int64)v47, *v47 + 9LL, 8);
          v44 = v74;
        }
        v46 += 8;
      }
      while ( v45 != v46 );
    }
  }
  else
  {
    v44 = v74;
  }
  _libc_free(v44);
  return v65;
}
