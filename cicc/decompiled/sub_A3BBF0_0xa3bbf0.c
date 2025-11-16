// Function: sub_A3BBF0
// Address: 0xa3bbf0
//
__int64 __fastcall sub_A3BBF0(__int64 *a1, __int64 a2, __m128i *a3, __int64 a4)
{
  __int64 v4; // r8
  __int64 v5; // rdi
  __int64 v6; // r12
  unsigned __int64 *v7; // rdx
  __int64 v8; // r8
  unsigned __int64 *v9; // r13
  unsigned __int64 *v10; // r15
  unsigned __int64 v11; // rdx
  unsigned __int64 *v12; // r13
  __int64 *v13; // r12
  __int64 *i; // r15
  __int64 v15; // rdx
  __int64 v16; // r12
  __int64 v17; // rax
  volatile signed __int32 *v18; // rax
  __int64 v19; // rdi
  volatile signed __int32 *v20; // rax
  volatile signed __int32 *v21; // r8
  __int64 v22; // rax
  volatile signed __int32 *v23; // rax
  __int64 v24; // rdi
  volatile signed __int32 *v25; // rax
  volatile signed __int32 *v26; // r8
  __int64 v27; // rax
  volatile signed __int32 *v28; // rax
  __int64 v29; // rdi
  volatile signed __int32 *v30; // rax
  volatile signed __int32 *v31; // r8
  __int64 v32; // rax
  __int64 *m128i_i64; // rsi
  volatile signed __int32 *v34; // rax
  __m128i *v35; // r13
  __int64 v36; // r12
  __int64 **v37; // rdx
  __int64 *v38; // rbx
  __int64 v39; // r15
  __int64 v40; // r13
  unsigned int v41; // eax
  int v42; // eax
  int v44; // edx
  __int64 ***v45; // rax
  __int64 **v46; // rcx
  __int64 ***v47; // rbx
  __int64 ***v48; // rcx
  __m128i *v49; // rdx
  __int64 **v50; // rdx
  __int64 ***v51; // rax
  __int64 **v52; // rdx
  __m128i *v53; // r12
  __int64 v54; // rbx
  unsigned __int64 v55; // rax
  __m128i *v56; // r12
  __int64 **v57; // rax
  __int64 v58; // rbx
  unsigned int v59; // eax
  int v60; // eax
  __m128i *v61; // rbx
  __m128i *v62; // rdi
  __int64 v63; // [rsp+10h] [rbp-290h]
  __int64 *v64; // [rsp+18h] [rbp-288h]
  __int64 *v65; // [rsp+18h] [rbp-288h]
  __int64 ***v66; // [rsp+18h] [rbp-288h]
  __int64 v67; // [rsp+18h] [rbp-288h]
  int v68; // [rsp+20h] [rbp-280h] BYREF
  int v69; // [rsp+24h] [rbp-27Ch] BYREF
  int v70; // [rsp+28h] [rbp-278h] BYREF
  int v71; // [rsp+2Ch] [rbp-274h] BYREF
  __int64 v72; // [rsp+30h] [rbp-270h] BYREF
  volatile signed __int32 *v73; // [rsp+38h] [rbp-268h]
  __m128i v74; // [rsp+40h] [rbp-260h] BYREF
  void *src; // [rsp+50h] [rbp-250h] BYREF
  __m128i *v76; // [rsp+58h] [rbp-248h]
  __m128i *v77; // [rsp+60h] [rbp-240h]
  __int64 *v78; // [rsp+70h] [rbp-230h] BYREF
  int *v79; // [rsp+78h] [rbp-228h]
  int *v80; // [rsp+80h] [rbp-220h]
  __int64 *v81; // [rsp+88h] [rbp-218h]
  __int64 *v82; // [rsp+90h] [rbp-210h]
  int *v83; // [rsp+98h] [rbp-208h]
  __int64 v84[2]; // [rsp+A0h] [rbp-200h] BYREF
  __int64 v85; // [rsp+B0h] [rbp-1F0h]
  __int64 v86; // [rsp+B8h] [rbp-1E8h]
  __m128i *v87; // [rsp+C0h] [rbp-1E0h]
  int v88; // [rsp+D0h] [rbp-1D0h] BYREF
  __int64 v89; // [rsp+D8h] [rbp-1C8h]
  int *v90; // [rsp+E0h] [rbp-1C0h]
  int *v91; // [rsp+E8h] [rbp-1B8h]
  __int64 v92; // [rsp+F0h] [rbp-1B0h]
  __int64 v93; // [rsp+F8h] [rbp-1A8h]
  __int64 v94; // [rsp+100h] [rbp-1A0h]
  __int64 v95; // [rsp+108h] [rbp-198h]
  __int64 v96; // [rsp+110h] [rbp-190h]
  __int64 v97; // [rsp+118h] [rbp-188h]
  __int64 v98; // [rsp+120h] [rbp-180h]
  unsigned int v99; // [rsp+128h] [rbp-178h]
  int v100; // [rsp+130h] [rbp-170h]
  __int64 v101; // [rsp+138h] [rbp-168h]
  __int64 v102; // [rsp+140h] [rbp-160h]
  __int64 v103; // [rsp+148h] [rbp-158h]
  unsigned int v104; // [rsp+150h] [rbp-150h]
  __int64 *v105; // [rsp+160h] [rbp-140h] BYREF
  __int64 v106; // [rsp+168h] [rbp-138h]
  _BYTE v107[304]; // [rsp+170h] [rbp-130h] BYREF

  v4 = (__int64)(a1 + 1);
  v5 = *a1;
  v87 = a3;
  v84[1] = v4;
  v84[0] = v5;
  v85 = a2;
  v86 = a4;
  v88 = 0;
  v89 = 0;
  v90 = &v88;
  v91 = &v88;
  v92 = 0;
  v93 = 0;
  v94 = 0;
  v95 = 0;
  v96 = 0;
  v97 = 0;
  v98 = 0;
  v99 = 0;
  v100 = 0;
  v101 = 0;
  v102 = 0;
  v103 = 0;
  v104 = 0;
  v78 = v84;
  v79 = (int *)a2;
  v105 = v84;
  v106 = (__int64)&v78;
  if ( !a3 )
  {
    v12 = *(unsigned __int64 **)(a2 + 24);
    if ( v12 == (unsigned __int64 *)(a2 + 8) )
      goto LABEL_22;
    do
    {
      v13 = (__int64 *)v12[8];
      for ( i = (__int64 *)v12[7]; v13 != i; ++i )
      {
        v15 = *i;
        sub_A31250(&v105, v12[4], v15, 0);
      }
      v12 = (unsigned __int64 *)sub_220EF30(v12);
    }
    while ( (unsigned __int64 *)(a2 + 8) != v12 );
    goto LABEL_21;
  }
  v6 = a3[1].m128i_i64[1];
  v64 = &a3->m128i_i64[1];
  if ( (unsigned __int64 *)v6 != &a3->m128i_u64[1] )
  {
    do
    {
      if ( *(_DWORD *)(v6 + 80) )
      {
        v7 = *(unsigned __int64 **)(v6 + 72);
        v8 = 2LL * *(unsigned int *)(v6 + 88);
        v9 = &v7[v8];
        if ( v7 != &v7[v8] )
        {
          while ( 1 )
          {
            v10 = v7;
            if ( *v7 <= 0xFFFFFFFFFFFFFFFDLL )
              break;
            v7 += 2;
            if ( v9 == v7 )
              goto LABEL_4;
          }
          while ( v9 != v10 )
          {
            sub_A31250(&v105, *v10, v10[1], 0);
            v11 = v10[1];
            if ( !*(_DWORD *)(v11 + 8) )
              sub_A31250(&v105, *(_QWORD *)(*(_QWORD *)(v11 + 56) & 0xFFFFFFFFFFFFFFF8LL), *(_QWORD *)(v11 + 64), 1);
            v10 += 2;
            if ( v10 == v9 )
              break;
            while ( *v10 > 0xFFFFFFFFFFFFFFFDLL )
            {
              v10 += 2;
              if ( v9 == v10 )
                goto LABEL_4;
            }
          }
        }
      }
LABEL_4:
      v6 = sub_220EF30(v6);
    }
    while ( v64 != (__int64 *)v6 );
LABEL_21:
    v5 = v84[0];
  }
LABEL_22:
  sub_A19830(v5, 8u, 3u);
  v16 = v84[0];
  sub_A17B10(v84[0], 3u, *(_DWORD *)(v84[0] + 56));
  sub_A17CC0(v16, 1u, 6);
  sub_A17CC0(v16, 1u, 6);
  sub_A17CC0(v16, 2u, 6);
  sub_A19830(v84[0], 0x13u, 3u);
  sub_A23770(&v72);
  sub_A186C0(v72, 1, 1);
  sub_A186C0(v72, 8, 4);
  sub_A186C0(v72, 0, 6);
  sub_A186C0(v72, 8, 2);
  v17 = v72;
  v72 = 0;
  v105 = (__int64 *)v17;
  v18 = v73;
  v73 = 0;
  v106 = (__int64)v18;
  v68 = sub_A1AB30(v84[0], (__int64 *)&v105);
  if ( v106 )
    sub_A191D0((volatile signed __int32 *)v106);
  sub_A23770(&v105);
  v19 = (__int64)v105;
  v20 = (volatile signed __int32 *)v106;
  v105 = 0;
  v21 = v73;
  v106 = 0;
  v72 = v19;
  v73 = v20;
  if ( v21 )
  {
    sub_A191D0(v21);
    if ( v106 )
      sub_A191D0((volatile signed __int32 *)v106);
    v19 = v72;
  }
  sub_A186C0(v19, 1, 1);
  sub_A186C0(v72, 8, 4);
  sub_A186C0(v72, 0, 6);
  sub_A186C0(v72, 7, 2);
  v22 = v72;
  v72 = 0;
  v105 = (__int64 *)v22;
  v23 = v73;
  v73 = 0;
  v106 = (__int64)v23;
  v69 = sub_A1AB30(v84[0], (__int64 *)&v105);
  if ( v106 )
    sub_A191D0((volatile signed __int32 *)v106);
  sub_A23770(&v105);
  v24 = (__int64)v105;
  v25 = (volatile signed __int32 *)v106;
  v105 = 0;
  v26 = v73;
  v106 = 0;
  v72 = v24;
  v73 = v25;
  if ( v26 )
  {
    sub_A191D0(v26);
    if ( v106 )
      sub_A191D0((volatile signed __int32 *)v106);
    v24 = v72;
  }
  sub_A186C0(v24, 1, 1);
  sub_A186C0(v72, 8, 4);
  sub_A186C0(v72, 0, 6);
  sub_A186C0(v72, 0, 8);
  v27 = v72;
  v72 = 0;
  v105 = (__int64 *)v27;
  v28 = v73;
  v73 = 0;
  v106 = (__int64)v28;
  v70 = sub_A1AB30(v84[0], (__int64 *)&v105);
  if ( v106 )
    sub_A191D0((volatile signed __int32 *)v106);
  sub_A23770(&v105);
  v29 = (__int64)v105;
  v30 = (volatile signed __int32 *)v106;
  v105 = 0;
  v31 = v73;
  v106 = 0;
  v72 = v29;
  v73 = v30;
  if ( v31 )
  {
    sub_A191D0(v31);
    if ( v106 )
      sub_A191D0((volatile signed __int32 *)v106);
    v29 = v72;
  }
  sub_A186C0(v29, 2, 1);
  sub_A186C0(v72, 32, 2);
  sub_A186C0(v72, 32, 2);
  sub_A186C0(v72, 32, 2);
  sub_A186C0(v72, 32, 2);
  sub_A186C0(v72, 32, 2);
  v32 = v72;
  m128i_i64 = (__int64 *)&v105;
  v72 = 0;
  v105 = (__int64 *)v32;
  v34 = v73;
  v73 = 0;
  v106 = (__int64)v34;
  v71 = sub_A1AB30(v84[0], (__int64 *)&v105);
  if ( v106 )
    sub_A191D0((volatile signed __int32 *)v106);
  v35 = v87;
  v82 = (__int64 *)&v105;
  v105 = (__int64 *)v107;
  v106 = 0x4000000000LL;
  v78 = (__int64 *)&v68;
  v79 = &v70;
  v80 = &v69;
  v81 = v84;
  v83 = &v71;
  if ( v87 )
  {
    v36 = v87[1].m128i_i64[1];
    v65 = &v87->m128i_i64[1];
    if ( (unsigned __int64 *)v36 != &v87->m128i_u64[1] )
    {
      do
      {
        v38 = *(__int64 **)(v36 + 32);
        v39 = *(_QWORD *)(v36 + 40);
        v40 = v85;
        v41 = sub_C92610(v38, v39);
        m128i_i64 = v38;
        v42 = sub_C92860(v40 + 48, v38, v39, v41);
        if ( v42 == -1 )
          v37 = (__int64 **)(*(_QWORD *)(v40 + 48) + 8LL * *(unsigned int *)(v40 + 56));
        else
          v37 = (__int64 **)(*(_QWORD *)(v40 + 48) + 8LL * v42);
        if ( v37 != (__int64 **)(*(_QWORD *)(v85 + 48) + 8LL * *(unsigned int *)(v85 + 56)) )
        {
          m128i_i64 = *v37;
          sub_A2B500((__int64)&v78, *v37);
        }
        v36 = sub_220EF30(v36);
      }
      while ( v65 != (__int64 *)v36 );
    }
  }
  else
  {
    src = 0;
    v76 = 0;
    v77 = 0;
    v44 = *(_DWORD *)(v85 + 56);
    if ( v44 )
    {
      v45 = *(__int64 ****)(v85 + 48);
      v46 = *v45;
      v47 = v45;
      if ( *v45 != (__int64 **)-8LL )
        goto LABEL_64;
      do
      {
        do
        {
          v46 = v47[1];
          ++v47;
        }
        while ( v46 == (__int64 **)-8LL );
LABEL_64:
        ;
      }
      while ( !v46 );
      v48 = &v45[v44];
      if ( v48 != v47 )
      {
        v49 = 0;
        while ( 1 )
        {
          m128i_i64 = **v47;
          v74.m128i_i64[0] = (__int64)(*v47 + 4);
          v74.m128i_i64[1] = (__int64)m128i_i64;
          if ( v49 == v35 )
          {
            m128i_i64 = (__int64 *)v35;
            v66 = v48;
            sub_A04210((const __m128i **)&src, v35, &v74);
            v35 = v76;
            v48 = v66;
          }
          else
          {
            if ( v35 )
            {
              *v35 = _mm_loadu_si128(&v74);
              v35 = v76;
            }
            v76 = ++v35;
          }
          v50 = v47[1];
          v51 = v47 + 1;
          if ( v50 == (__int64 **)-8LL || !v50 )
          {
            do
            {
              do
              {
                v52 = v51[1];
                ++v51;
              }
              while ( !v52 );
            }
            while ( v52 == (__int64 **)-8LL );
          }
          if ( v48 == v51 )
            break;
          v49 = v77;
          v47 = v51;
        }
        v53 = (__m128i *)src;
        if ( src != v35 )
        {
          v54 = (char *)v35 - (_BYTE *)src;
          _BitScanReverse64(&v55, ((char *)v35 - (_BYTE *)src) >> 4);
          sub_A3B9A0((__m128i *)src, v35, 2LL * (int)(63 - (v55 ^ 0x3F)));
          if ( v54 > 256 )
          {
            v61 = v53 + 16;
            m128i_i64 = v53[16].m128i_i64;
            sub_A3B670(v53, v53 + 16);
            if ( &v53[16] != v35 )
            {
              do
              {
                v62 = v61++;
                sub_A3B600(v62);
              }
              while ( v35 != v61 );
            }
          }
          else
          {
            m128i_i64 = (__int64 *)v35;
            sub_A3B670(v53, v35);
          }
          v35 = v76;
          if ( src != v76 )
          {
            v56 = (__m128i *)src;
            do
            {
              v58 = v85;
              v63 = v56->m128i_i64[0];
              v67 = v56->m128i_i64[1];
              v59 = sub_C92610(v56->m128i_i64[0], v67);
              v60 = sub_C92860(v58 + 48, v63, v67, v59);
              if ( v60 == -1 )
                v57 = (__int64 **)(*(_QWORD *)(v58 + 48) + 8LL * *(unsigned int *)(v58 + 56));
              else
                v57 = (__int64 **)(*(_QWORD *)(v58 + 48) + 8LL * v60);
              m128i_i64 = *v57;
              ++v56;
              sub_A2B500((__int64)&v78, *v57);
            }
            while ( v35 != v56 );
            v35 = (__m128i *)src;
          }
        }
        if ( v35 )
        {
          m128i_i64 = (__int64 *)((char *)v77 - (char *)v35);
          j_j___libc_free_0(v35, (char *)v77 - (char *)v35);
        }
      }
    }
  }
  sub_A192A0(v84[0]);
  if ( v105 != (__int64 *)v107 )
    _libc_free(v105, m128i_i64);
  if ( v73 )
    sub_A191D0(v73);
  sub_A33C40((__int64)v84);
  sub_A192A0(v84[0]);
  sub_C7D6A0(v102, 24LL * v104, 8);
  sub_C7D6A0(v97, 8LL * v99, 4);
  if ( v93 )
    j_j___libc_free_0(v93, v95 - v93);
  return sub_A167C0(v89);
}
