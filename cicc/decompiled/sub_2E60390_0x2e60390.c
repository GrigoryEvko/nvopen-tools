// Function: sub_2E60390
// Address: 0x2e60390
//
__int64 *__fastcall sub_2E60390(__int64 a1, __int64 a2)
{
  __int64 *result; // rax
  __int64 v4; // rdx
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // rax
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  unsigned __int64 v15; // rax
  unsigned __int64 v16; // rax
  unsigned __int64 v17; // rax
  __int64 v18; // r8
  __int64 v19; // r9
  unsigned __int64 v20; // rsi
  const __m128i *v21; // rdi
  unsigned __int64 v22; // rdx
  __int64 v23; // rax
  unsigned __int64 v24; // rdx
  __int64 v25; // rcx
  __m128i *v26; // rdx
  const __m128i *v27; // rax
  const __m128i *v28; // rcx
  __int64 v29; // rax
  unsigned __int64 v30; // rdi
  __m128i *v31; // rdx
  const __m128i *v32; // rax
  unsigned __int64 v33; // rcx
  unsigned __int64 v34; // rax
  __int64 v35; // r15
  __int64 v36; // r13
  unsigned int i; // r12d
  _DWORD *v38; // rdx
  __int64 v39; // rcx
  __int64 v40; // r8
  __int64 v41; // r9
  _BYTE *v42; // rax
  unsigned __int64 v43; // r13
  __int64 v44; // r14
  __int64 *v45; // rax
  __int64 *v46; // rdx
  __int64 v47; // r12
  __int64 *v48; // rax
  char v49; // dl
  unsigned __int64 v50; // rdx
  char v51; // si
  unsigned __int64 v52; // [rsp+0h] [rbp-3A0h]
  unsigned __int64 v53; // [rsp+0h] [rbp-3A0h]
  __int64 *v54; // [rsp+30h] [rbp-370h]
  __int64 *v55; // [rsp+68h] [rbp-338h]
  unsigned __int64 v57[16]; // [rsp+80h] [rbp-320h] BYREF
  __m128i v58; // [rsp+100h] [rbp-2A0h] BYREF
  __int64 v59; // [rsp+110h] [rbp-290h]
  _QWORD *(__fastcall *v60)(__int64 *, __int64); // [rsp+118h] [rbp-288h]
  _QWORD v61[8]; // [rsp+120h] [rbp-280h] BYREF
  unsigned __int64 v62; // [rsp+160h] [rbp-240h] BYREF
  unsigned __int64 v63; // [rsp+168h] [rbp-238h]
  unsigned __int64 v64; // [rsp+170h] [rbp-230h]
  __int64 v65; // [rsp+180h] [rbp-220h] BYREF
  __int64 *v66; // [rsp+188h] [rbp-218h]
  unsigned int v67; // [rsp+190h] [rbp-210h]
  unsigned int v68; // [rsp+194h] [rbp-20Ch]
  char v69; // [rsp+19Ch] [rbp-204h]
  _BYTE v70[64]; // [rsp+1A0h] [rbp-200h] BYREF
  unsigned __int64 v71; // [rsp+1E0h] [rbp-1C0h] BYREF
  unsigned __int64 v72; // [rsp+1E8h] [rbp-1B8h]
  unsigned __int64 v73; // [rsp+1F0h] [rbp-1B0h]
  char v74[8]; // [rsp+200h] [rbp-1A0h] BYREF
  unsigned __int64 v75; // [rsp+208h] [rbp-198h]
  char v76; // [rsp+21Ch] [rbp-184h]
  _BYTE v77[64]; // [rsp+220h] [rbp-180h] BYREF
  unsigned __int64 v78; // [rsp+260h] [rbp-140h]
  unsigned __int64 v79; // [rsp+268h] [rbp-138h]
  unsigned __int64 v80; // [rsp+270h] [rbp-130h]
  __m128i v81; // [rsp+280h] [rbp-120h] BYREF
  char v82; // [rsp+290h] [rbp-110h]
  char v83; // [rsp+29Ch] [rbp-104h]
  char v84[64]; // [rsp+2A0h] [rbp-100h] BYREF
  const __m128i *v85; // [rsp+2E0h] [rbp-C0h]
  unsigned __int64 v86; // [rsp+2E8h] [rbp-B8h]
  unsigned __int64 v87; // [rsp+2F0h] [rbp-B0h]
  char v88[8]; // [rsp+2F8h] [rbp-A8h] BYREF
  unsigned __int64 v89; // [rsp+300h] [rbp-A0h]
  char v90; // [rsp+314h] [rbp-8Ch]
  char v91[64]; // [rsp+318h] [rbp-88h] BYREF
  const __m128i *v92; // [rsp+358h] [rbp-48h]
  const __m128i *v93; // [rsp+360h] [rbp-40h]
  unsigned __int64 v94; // [rsp+368h] [rbp-38h]

  result = *(__int64 **)(a1 + 72);
  v54 = *(__int64 **)(a1 + 80);
  if ( result != v54 )
  {
    v55 = *(__int64 **)(a1 + 72);
    do
    {
      v4 = *v55;
      v62 = 0;
      memset(v57, 0, 0x78u);
      v57[1] = (unsigned __int64)&v57[4];
      v59 = 0x100000008LL;
      v61[0] = v4;
      v81.m128i_i64[0] = v4;
      LODWORD(v57[2]) = 8;
      BYTE4(v57[3]) = 1;
      v58.m128i_i64[1] = (__int64)v61;
      v63 = 0;
      v64 = 0;
      LODWORD(v60) = 0;
      BYTE4(v60) = 1;
      v58.m128i_i64[0] = 1;
      v82 = 0;
      sub_2E60350(&v62, &v81);
      sub_C8CF70((__int64)v74, v77, 8, (__int64)&v57[4], (__int64)v57);
      v5 = v57[12];
      memset(&v57[12], 0, 24);
      v78 = v5;
      v79 = v57[13];
      v80 = v57[14];
      sub_C8CF70((__int64)&v65, v70, 8, (__int64)v61, (__int64)&v58);
      v6 = v62;
      v62 = 0;
      v71 = v6;
      v7 = v63;
      v63 = 0;
      v72 = v7;
      v8 = v64;
      v64 = 0;
      v73 = v8;
      sub_C8CF70((__int64)&v81, v84, 8, (__int64)v70, (__int64)&v65);
      v9 = v71;
      v71 = 0;
      v85 = (const __m128i *)v9;
      v10 = v72;
      v72 = 0;
      v86 = v10;
      v11 = v73;
      v73 = 0;
      v87 = v11;
      sub_C8CF70((__int64)v88, v91, 8, (__int64)v77, (__int64)v74);
      v15 = v78;
      v78 = 0;
      v92 = (const __m128i *)v15;
      v16 = v79;
      v79 = 0;
      v93 = (const __m128i *)v16;
      v17 = v80;
      v80 = 0;
      v94 = v17;
      if ( v71 )
        j_j___libc_free_0(v71);
      if ( !v69 )
        _libc_free((unsigned __int64)v66);
      if ( v78 )
        j_j___libc_free_0(v78);
      if ( !v76 )
        _libc_free(v75);
      if ( v62 )
        j_j___libc_free_0(v62);
      if ( !BYTE4(v60) )
        _libc_free(v58.m128i_u64[1]);
      if ( v57[12] )
        j_j___libc_free_0(v57[12]);
      if ( !BYTE4(v57[3]) )
        _libc_free(v57[1]);
      sub_C8CD80((__int64)&v65, (__int64)v70, (__int64)&v81, v12, v13, v14);
      v20 = v86;
      v21 = v85;
      v71 = 0;
      v72 = 0;
      v73 = 0;
      v22 = v86 - (_QWORD)v85;
      if ( (const __m128i *)v86 == v85 )
      {
        v24 = 0;
        v25 = 0;
      }
      else
      {
        if ( v22 > 0x7FFFFFFFFFFFFFF8LL )
          goto LABEL_91;
        v52 = v86 - (_QWORD)v85;
        v23 = sub_22077B0(v86 - (_QWORD)v85);
        v20 = v86;
        v21 = v85;
        v24 = v52;
        v25 = v23;
      }
      v71 = v25;
      v72 = v25;
      v73 = v25 + v24;
      if ( v21 != (const __m128i *)v20 )
      {
        v26 = (__m128i *)v25;
        v27 = v21;
        do
        {
          if ( v26 )
          {
            *v26 = _mm_loadu_si128(v27);
            v18 = v27[1].m128i_i64[0];
            v26[1].m128i_i64[0] = v18;
          }
          v27 = (const __m128i *)((char *)v27 + 24);
          v26 = (__m128i *)((char *)v26 + 24);
        }
        while ( v27 != (const __m128i *)v20 );
        v25 += 8 * ((unsigned __int64)((char *)&v27[-2].m128i_u64[1] - (char *)v21) >> 3) + 24;
      }
      v72 = v25;
      v21 = (const __m128i *)v74;
      sub_C8CD80((__int64)v74, (__int64)v77, (__int64)v88, v25, v18, v19);
      v28 = v93;
      v20 = (unsigned __int64)v92;
      v78 = 0;
      v79 = 0;
      v80 = 0;
      v22 = (char *)v93 - (char *)v92;
      if ( v93 == v92 )
      {
        v30 = 0;
      }
      else
      {
        if ( v22 > 0x7FFFFFFFFFFFFFF8LL )
LABEL_91:
          sub_4261EA(v21, v20, v22);
        v53 = (char *)v93 - (char *)v92;
        v29 = sub_22077B0((char *)v93 - (char *)v92);
        v28 = v93;
        v20 = (unsigned __int64)v92;
        v22 = v53;
        v30 = v29;
      }
      v78 = v30;
      v80 = v30 + v22;
      v31 = (__m128i *)v30;
      v79 = v30;
      if ( v28 != (const __m128i *)v20 )
      {
        v32 = (const __m128i *)v20;
        do
        {
          if ( v31 )
          {
            *v31 = _mm_loadu_si128(v32);
            v31[1].m128i_i64[0] = v32[1].m128i_i64[0];
          }
          v32 = (const __m128i *)((char *)v32 + 24);
          v31 = (__m128i *)((char *)v31 + 24);
        }
        while ( v28 != v32 );
        v31 = (__m128i *)(v30 + 8 * (((unsigned __int64)&v28[-2].m128i_u64[1] - v20) >> 3) + 24);
      }
      v33 = v72;
      v34 = v71;
      v79 = (unsigned __int64)v31;
      v35 = a2;
      if ( (__m128i *)(v72 - v71) == (__m128i *)((char *)v31 - v30) )
        goto LABEL_61;
      do
      {
LABEL_38:
        v36 = *(_QWORD *)(v33 - 24);
        for ( i = 0; *(_DWORD *)(v36 + 168) > i; ++i )
        {
          while ( 1 )
          {
            v38 = *(_DWORD **)(v35 + 32);
            if ( *(_QWORD *)(v35 + 24) - (_QWORD)v38 <= 3u )
              break;
            *v38 = 538976288;
            ++i;
            *(_QWORD *)(v35 + 32) += 4LL;
            if ( *(_DWORD *)(v36 + 168) <= i )
              goto LABEL_43;
          }
          sub_CB6200(v35, (unsigned __int8 *)"    ", 4u);
        }
LABEL_43:
        v58.m128i_i64[0] = v36;
        v58.m128i_i64[1] = a1;
        v59 = (__int64)sub_2E5D7C0;
        v60 = sub_2E5F810;
        sub_2E5F810(v58.m128i_i64, v35);
        v42 = *(_BYTE **)(v35 + 32);
        if ( (unsigned __int64)v42 >= *(_QWORD *)(v35 + 24) )
        {
          sub_CB5D20(v35, 10);
        }
        else
        {
          *(_QWORD *)(v35 + 32) = v42 + 1;
          *v42 = 10;
        }
        if ( v59 )
          ((void (__fastcall *)(__m128i *, __m128i *, __int64))v59)(&v58, &v58, 3);
        v43 = v72;
        while ( 1 )
        {
          v44 = *(_QWORD *)(v43 - 24);
          if ( *(_BYTE *)(v43 - 8) )
            break;
          v45 = *(__int64 **)(v44 + 32);
          *(_BYTE *)(v43 - 8) = 1;
          *(_QWORD *)(v43 - 16) = v45;
          if ( v45 != *(__int64 **)(v44 + 40) )
            goto LABEL_50;
LABEL_56:
          v72 -= 24LL;
          v34 = v71;
          v43 = v72;
          if ( v72 == v71 )
          {
            v33 = v71;
            goto LABEL_60;
          }
        }
        while ( 1 )
        {
          while ( 1 )
          {
            v45 = *(__int64 **)(v43 - 16);
            if ( v45 == *(__int64 **)(v44 + 40) )
              goto LABEL_56;
LABEL_50:
            v46 = v45 + 1;
            *(_QWORD *)(v43 - 16) = v45 + 1;
            v47 = *v45;
            if ( v69 )
              break;
LABEL_58:
            sub_C8CC70((__int64)&v65, v47, (__int64)v46, v39, v40, v41);
            if ( v49 )
              goto LABEL_59;
          }
          v48 = v66;
          v46 = &v66[v68];
          if ( v66 == v46 )
            break;
          while ( v47 != *v48 )
          {
            if ( v46 == ++v48 )
              goto LABEL_86;
          }
        }
LABEL_86:
        if ( v68 >= v67 )
          goto LABEL_58;
        ++v68;
        *v46 = v47;
        ++v65;
LABEL_59:
        v58.m128i_i64[0] = v47;
        LOBYTE(v59) = 0;
        sub_2E60350(&v71, &v58);
        v34 = v71;
        v33 = v72;
LABEL_60:
        v30 = v78;
      }
      while ( v33 - v34 != v79 - v78 );
LABEL_61:
      if ( v33 != v34 )
      {
        v50 = v30;
        while ( *(_QWORD *)v34 == *(_QWORD *)v50 )
        {
          v51 = *(_BYTE *)(v34 + 16);
          if ( v51 != *(_BYTE *)(v50 + 16) || v51 && *(_QWORD *)(v34 + 8) != *(_QWORD *)(v50 + 8) )
            break;
          v34 += 24LL;
          v50 += 24LL;
          if ( v33 == v34 )
            goto LABEL_68;
        }
        goto LABEL_38;
      }
LABEL_68:
      a2 = v35;
      if ( v30 )
        j_j___libc_free_0(v30);
      if ( !v76 )
        _libc_free(v75);
      if ( v71 )
        j_j___libc_free_0(v71);
      if ( !v69 )
        _libc_free((unsigned __int64)v66);
      if ( v92 )
        j_j___libc_free_0((unsigned __int64)v92);
      if ( !v90 )
        _libc_free(v89);
      if ( v85 )
        j_j___libc_free_0((unsigned __int64)v85);
      if ( !v83 )
        _libc_free(v81.m128i_u64[1]);
      result = ++v55;
    }
    while ( v54 != v55 );
  }
  return result;
}
