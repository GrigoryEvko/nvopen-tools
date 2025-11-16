// Function: sub_E39A30
// Address: 0xe39a30
//
__int64 *__fastcall sub_E39A30(__int64 a1, __int64 a2)
{
  __int64 *result; // rax
  __int64 v4; // rdx
  __m128i *v5; // rax
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rax
  const __m128i *v9; // rax
  _BYTE *v10; // rsi
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  __m128i *v16; // rax
  __m128i *v17; // rax
  __int8 *v18; // rax
  __int64 v19; // r8
  __int64 v20; // r9
  const __m128i *v21; // rsi
  const __m128i *v22; // rdi
  unsigned __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rcx
  __m128i *v27; // rdx
  const __m128i *v28; // rax
  const __m128i *v29; // rcx
  __int64 v30; // rax
  __m128i *v31; // rdi
  __m128i *v32; // rdx
  const __m128i *v33; // rax
  __int64 v34; // rcx
  __int64 v35; // rax
  __int64 v36; // r15
  __int64 v37; // r13
  unsigned int i; // r12d
  _DWORD *v39; // rdx
  __int64 v40; // rcx
  __int64 v41; // r8
  __int64 v42; // r9
  _BYTE *v43; // rax
  __int64 v44; // r13
  __int64 v45; // r14
  __int64 *v46; // rax
  __int64 *v47; // rdx
  __int64 v48; // r12
  __int64 *v49; // rax
  char v50; // dl
  __m128i *v51; // rdx
  char v52; // si
  __int64 v53; // rsi
  signed __int64 v54; // [rsp+0h] [rbp-3A0h]
  unsigned __int64 v55; // [rsp+0h] [rbp-3A0h]
  __int64 *v56; // [rsp+30h] [rbp-370h]
  __int64 *v57; // [rsp+68h] [rbp-338h]
  _QWORD v59[16]; // [rsp+80h] [rbp-320h] BYREF
  __m128i v60; // [rsp+100h] [rbp-2A0h] BYREF
  __int64 v61; // [rsp+110h] [rbp-290h]
  _QWORD *(__fastcall *v62)(__int64 *, __int64); // [rsp+118h] [rbp-288h]
  _QWORD v63[8]; // [rsp+120h] [rbp-280h] BYREF
  __int64 v64; // [rsp+160h] [rbp-240h] BYREF
  __int64 v65; // [rsp+168h] [rbp-238h]
  __int64 v66; // [rsp+170h] [rbp-230h]
  __int64 v67; // [rsp+180h] [rbp-220h] BYREF
  __int64 *v68; // [rsp+188h] [rbp-218h]
  unsigned int v69; // [rsp+190h] [rbp-210h]
  unsigned int v70; // [rsp+194h] [rbp-20Ch]
  char v71; // [rsp+19Ch] [rbp-204h]
  _BYTE v72[64]; // [rsp+1A0h] [rbp-200h] BYREF
  __int64 v73; // [rsp+1E0h] [rbp-1C0h] BYREF
  __int64 v74; // [rsp+1E8h] [rbp-1B8h]
  __int64 v75; // [rsp+1F0h] [rbp-1B0h]
  char v76[8]; // [rsp+200h] [rbp-1A0h] BYREF
  __int64 v77; // [rsp+208h] [rbp-198h]
  char v78; // [rsp+21Ch] [rbp-184h]
  _BYTE v79[64]; // [rsp+220h] [rbp-180h] BYREF
  __m128i *v80; // [rsp+260h] [rbp-140h]
  __m128i *v81; // [rsp+268h] [rbp-138h]
  __int8 *v82; // [rsp+270h] [rbp-130h]
  __m128i v83; // [rsp+280h] [rbp-120h] BYREF
  char v84; // [rsp+290h] [rbp-110h]
  char v85; // [rsp+29Ch] [rbp-104h]
  char v86[64]; // [rsp+2A0h] [rbp-100h] BYREF
  const __m128i *v87; // [rsp+2E0h] [rbp-C0h]
  const __m128i *v88; // [rsp+2E8h] [rbp-B8h]
  __int64 v89; // [rsp+2F0h] [rbp-B0h]
  char v90[8]; // [rsp+2F8h] [rbp-A8h] BYREF
  __int64 v91; // [rsp+300h] [rbp-A0h]
  char v92; // [rsp+314h] [rbp-8Ch]
  _BYTE v93[64]; // [rsp+318h] [rbp-88h] BYREF
  const __m128i *v94; // [rsp+358h] [rbp-48h]
  const __m128i *v95; // [rsp+360h] [rbp-40h]
  __int8 *v96; // [rsp+368h] [rbp-38h]

  result = *(__int64 **)(a1 + 72);
  v56 = *(__int64 **)(a1 + 80);
  if ( result != v56 )
  {
    v57 = *(__int64 **)(a1 + 72);
    do
    {
      v4 = *v57;
      v64 = 0;
      memset(v59, 0, 0x78u);
      v59[1] = &v59[4];
      v61 = 0x100000008LL;
      v63[0] = v4;
      v83.m128i_i64[0] = v4;
      LODWORD(v59[2]) = 8;
      BYTE4(v59[3]) = 1;
      v60.m128i_i64[1] = (__int64)v63;
      v65 = 0;
      v66 = 0;
      LODWORD(v62) = 0;
      BYTE4(v62) = 1;
      v60.m128i_i64[0] = 1;
      v84 = 0;
      sub_E399F0((__int64)&v64, &v83);
      sub_C8CF70((__int64)v76, v79, 8, (__int64)&v59[4], (__int64)v59);
      v5 = (__m128i *)v59[12];
      memset(&v59[12], 0, 24);
      v80 = v5;
      v81 = (__m128i *)v59[13];
      v82 = (__int8 *)v59[14];
      sub_C8CF70((__int64)&v67, v72, 8, (__int64)v63, (__int64)&v60);
      v6 = v64;
      v64 = 0;
      v73 = v6;
      v7 = v65;
      v65 = 0;
      v74 = v7;
      v8 = v66;
      v66 = 0;
      v75 = v8;
      sub_C8CF70((__int64)&v83, v86, 8, (__int64)v72, (__int64)&v67);
      v9 = (const __m128i *)v73;
      v10 = v93;
      v73 = 0;
      v87 = v9;
      v11 = v74;
      v74 = 0;
      v88 = (const __m128i *)v11;
      v12 = v75;
      v75 = 0;
      v89 = v12;
      sub_C8CF70((__int64)v90, v93, 8, (__int64)v79, (__int64)v76);
      v16 = v80;
      v80 = 0;
      v94 = v16;
      v17 = v81;
      v81 = 0;
      v95 = v17;
      v18 = v82;
      v82 = 0;
      v96 = v18;
      if ( v73 )
      {
        v10 = (_BYTE *)(v75 - v73);
        j_j___libc_free_0(v73, v75 - v73);
      }
      if ( !v71 )
        _libc_free(v68, v10);
      if ( v80 )
      {
        v10 = (_BYTE *)(v82 - (__int8 *)v80);
        j_j___libc_free_0(v80, v82 - (__int8 *)v80);
      }
      if ( !v78 )
        _libc_free(v77, v10);
      if ( v64 )
      {
        v10 = (_BYTE *)(v66 - v64);
        j_j___libc_free_0(v64, v66 - v64);
      }
      if ( !BYTE4(v62) )
        _libc_free(v60.m128i_i64[1], v10);
      if ( v59[12] )
      {
        v10 = (_BYTE *)(v59[14] - v59[12]);
        j_j___libc_free_0(v59[12], v59[14] - v59[12]);
      }
      if ( !BYTE4(v59[3]) )
        _libc_free(v59[1], v10);
      sub_C8CD80((__int64)&v67, (__int64)v72, (__int64)&v83, v13, v14, v15);
      v21 = v88;
      v22 = v87;
      v73 = 0;
      v74 = 0;
      v75 = 0;
      v23 = (char *)v88 - (char *)v87;
      if ( v88 == v87 )
      {
        v25 = 0;
        v26 = 0;
      }
      else
      {
        if ( v23 > 0x7FFFFFFFFFFFFFF8LL )
          goto LABEL_91;
        v54 = (char *)v88 - (char *)v87;
        v24 = sub_22077B0((char *)v88 - (char *)v87);
        v21 = v88;
        v22 = v87;
        v25 = v54;
        v26 = v24;
      }
      v73 = v26;
      v74 = v26;
      v75 = v26 + v25;
      if ( v22 != v21 )
      {
        v27 = (__m128i *)v26;
        v28 = v22;
        do
        {
          if ( v27 )
          {
            *v27 = _mm_loadu_si128(v28);
            v19 = v28[1].m128i_i64[0];
            v27[1].m128i_i64[0] = v19;
          }
          v28 = (const __m128i *)((char *)v28 + 24);
          v27 = (__m128i *)((char *)v27 + 24);
        }
        while ( v28 != v21 );
        v26 += 8 * ((unsigned __int64)((char *)&v28[-2].m128i_u64[1] - (char *)v22) >> 3) + 24;
      }
      v74 = v26;
      v22 = (const __m128i *)v76;
      sub_C8CD80((__int64)v76, (__int64)v79, (__int64)v90, v26, v19, v20);
      v29 = v95;
      v21 = v94;
      v80 = 0;
      v81 = 0;
      v82 = 0;
      v23 = (char *)v95 - (char *)v94;
      if ( v95 == v94 )
      {
        v31 = 0;
      }
      else
      {
        if ( v23 > 0x7FFFFFFFFFFFFFF8LL )
LABEL_91:
          sub_4261EA(v22, v21, v23);
        v55 = (char *)v95 - (char *)v94;
        v30 = sub_22077B0((char *)v95 - (char *)v94);
        v29 = v95;
        v21 = v94;
        v23 = v55;
        v31 = (__m128i *)v30;
      }
      v80 = v31;
      v82 = &v31->m128i_i8[v23];
      v32 = v31;
      v81 = v31;
      if ( v29 != v21 )
      {
        v33 = v21;
        do
        {
          if ( v32 )
          {
            *v32 = _mm_loadu_si128(v33);
            v32[1].m128i_i64[0] = v33[1].m128i_i64[0];
          }
          v33 = (const __m128i *)((char *)v33 + 24);
          v32 = (__m128i *)((char *)v32 + 24);
        }
        while ( v29 != v33 );
        v32 = (__m128i *)((char *)v31 + 8 * ((unsigned __int64)((char *)&v29[-2].m128i_u64[1] - (char *)v21) >> 3) + 24);
      }
      v34 = v74;
      v35 = v73;
      v81 = v32;
      v36 = a2;
      if ( v74 - v73 == (char *)v32 - (char *)v31 )
        goto LABEL_61;
      do
      {
LABEL_38:
        v37 = *(_QWORD *)(v34 - 24);
        for ( i = 0; *(_DWORD *)(v37 + 168) > i; ++i )
        {
          while ( 1 )
          {
            v39 = *(_DWORD **)(v36 + 32);
            if ( *(_QWORD *)(v36 + 24) - (_QWORD)v39 <= 3u )
              break;
            *v39 = 538976288;
            ++i;
            *(_QWORD *)(v36 + 32) += 4LL;
            if ( *(_DWORD *)(v37 + 168) <= i )
              goto LABEL_43;
          }
          sub_CB6200(v36, (unsigned __int8 *)"    ", 4u);
        }
LABEL_43:
        v60.m128i_i64[0] = v37;
        v60.m128i_i64[1] = a1;
        v61 = (__int64)sub_E341B0;
        v62 = sub_E34BC0;
        sub_E34BC0(v60.m128i_i64, v36);
        v43 = *(_BYTE **)(v36 + 32);
        if ( (unsigned __int64)v43 >= *(_QWORD *)(v36 + 24) )
        {
          sub_CB5D20(v36, 10);
        }
        else
        {
          *(_QWORD *)(v36 + 32) = v43 + 1;
          *v43 = 10;
        }
        if ( v61 )
          ((void (__fastcall *)(__m128i *, __m128i *, __int64))v61)(&v60, &v60, 3);
        v44 = v74;
        while ( 1 )
        {
          v45 = *(_QWORD *)(v44 - 24);
          if ( *(_BYTE *)(v44 - 8) )
            break;
          v46 = *(__int64 **)(v45 + 32);
          *(_BYTE *)(v44 - 8) = 1;
          *(_QWORD *)(v44 - 16) = v46;
          if ( v46 != *(__int64 **)(v45 + 40) )
            goto LABEL_50;
LABEL_56:
          v74 -= 24;
          v35 = v73;
          v44 = v74;
          if ( v74 == v73 )
          {
            v34 = v73;
            goto LABEL_60;
          }
        }
        while ( 1 )
        {
          while ( 1 )
          {
            v46 = *(__int64 **)(v44 - 16);
            if ( v46 == *(__int64 **)(v45 + 40) )
              goto LABEL_56;
LABEL_50:
            v47 = v46 + 1;
            *(_QWORD *)(v44 - 16) = v46 + 1;
            v48 = *v46;
            if ( v71 )
              break;
LABEL_58:
            sub_C8CC70((__int64)&v67, v48, (__int64)v47, v40, v41, v42);
            if ( v50 )
              goto LABEL_59;
          }
          v49 = v68;
          v47 = &v68[v70];
          if ( v68 == v47 )
            break;
          while ( v48 != *v49 )
          {
            if ( v47 == ++v49 )
              goto LABEL_86;
          }
        }
LABEL_86:
        if ( v70 >= v69 )
          goto LABEL_58;
        ++v70;
        *v47 = v48;
        ++v67;
LABEL_59:
        v60.m128i_i64[0] = v48;
        LOBYTE(v61) = 0;
        sub_E399F0((__int64)&v73, &v60);
        v35 = v73;
        v34 = v74;
LABEL_60:
        v31 = v80;
      }
      while ( v34 - v35 != (char *)v81 - (char *)v80 );
LABEL_61:
      if ( v34 != v35 )
      {
        v51 = v31;
        while ( *(_QWORD *)v35 == v51->m128i_i64[0] )
        {
          v52 = *(_BYTE *)(v35 + 16);
          if ( v52 != v51[1].m128i_i8[0] || v52 && *(_QWORD *)(v35 + 8) != v51->m128i_i64[1] )
            break;
          v35 += 24;
          v51 = (__m128i *)((char *)v51 + 24);
          if ( v34 == v35 )
            goto LABEL_68;
        }
        goto LABEL_38;
      }
LABEL_68:
      a2 = v36;
      v53 = v82 - (__int8 *)v31;
      if ( v31 )
        j_j___libc_free_0(v31, v53);
      if ( !v78 )
        _libc_free(v77, v53);
      if ( v73 )
      {
        v53 = v75 - v73;
        j_j___libc_free_0(v73, v75 - v73);
      }
      if ( !v71 )
        _libc_free(v68, v53);
      if ( v94 )
      {
        v53 = v96 - (__int8 *)v94;
        j_j___libc_free_0(v94, v96 - (__int8 *)v94);
      }
      if ( !v92 )
        _libc_free(v91, v53);
      if ( v87 )
      {
        v53 = v89 - (_QWORD)v87;
        j_j___libc_free_0(v87, v89 - (_QWORD)v87);
      }
      if ( !v85 )
        _libc_free(v83.m128i_i64[1], v53);
      result = ++v57;
    }
    while ( v56 != v57 );
  }
  return result;
}
