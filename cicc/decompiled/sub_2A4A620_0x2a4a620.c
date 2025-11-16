// Function: sub_2A4A620
// Address: 0x2a4a620
//
void __fastcall sub_2A4A620(__int64 a1)
{
  __int64 v2; // rbx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 *v5; // rdi
  unsigned int v6; // r13d
  unsigned __int32 v7; // eax
  __int64 v8; // r8
  __int64 v9; // rax
  __int64 v10; // r10
  unsigned __int64 v11; // rdx
  __int64 *v12; // rax
  __int64 *v13; // rcx
  __int64 *v14; // rdx
  __int64 v15; // rdx
  unsigned __int64 v16; // rax
  unsigned __int64 v17; // rax
  unsigned __int64 v18; // rax
  unsigned __int64 v19; // rax
  unsigned __int64 v20; // rax
  unsigned __int64 v21; // rax
  unsigned __int64 v22; // rax
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // r9
  unsigned __int64 v26; // rax
  unsigned __int64 v27; // rax
  unsigned __int64 v28; // rax
  const __m128i *v29; // rsi
  __int64 v30; // rdx
  __int64 v31; // r8
  __int64 v32; // r9
  __int64 v33; // rcx
  const __m128i *v34; // rdi
  unsigned __int64 v35; // r14
  __int64 v36; // rax
  unsigned __int64 v37; // rsi
  __m128i *v38; // rdx
  const __m128i *v39; // rax
  __int64 *v40; // r8
  __int64 *v41; // r9
  const __m128i *v42; // rcx
  unsigned __int64 v43; // r14
  __int64 v44; // rax
  unsigned __int64 v45; // rdi
  __m128i *v46; // rdx
  const __m128i *v47; // rax
  unsigned __int64 v48; // r14
  unsigned __int64 v49; // rax
  unsigned __int64 *v50; // rcx
  __int64 v51; // rdx
  __int64 v52; // r13
  __int64 v53; // rdx
  unsigned __int64 v54; // rax
  char v55; // si
  __int64 *v56; // rax
  __int64 *v57; // rdx
  __int64 v58; // r15
  __int64 *v59; // rax
  char v60; // dl
  char v61; // si
  __int64 v62; // rsi
  __int64 v63; // r13
  __int64 v64; // rbx
  __int64 v65; // r13
  __int64 v66; // rsi
  __int64 v67; // rax
  __int64 v68; // rdx
  __int64 v69; // rcx
  __int64 v70; // rdi
  unsigned int v71; // eax
  __int64 v72; // r9
  int v73; // [rsp+8h] [rbp-4A8h]
  __int64 v74; // [rsp+10h] [rbp-4A0h]
  __int64 v75; // [rsp+18h] [rbp-498h]
  unsigned __int64 v76[2]; // [rsp+20h] [rbp-490h] BYREF
  _BYTE v77[64]; // [rsp+30h] [rbp-480h] BYREF
  unsigned __int64 v78[16]; // [rsp+70h] [rbp-440h] BYREF
  __m128i v79; // [rsp+F0h] [rbp-3C0h] BYREF
  __int64 v80; // [rsp+100h] [rbp-3B0h]
  int v81; // [rsp+108h] [rbp-3A8h]
  char v82; // [rsp+10Ch] [rbp-3A4h]
  _QWORD v83[8]; // [rsp+110h] [rbp-3A0h] BYREF
  unsigned __int64 v84; // [rsp+150h] [rbp-360h] BYREF
  unsigned __int64 v85; // [rsp+158h] [rbp-358h]
  unsigned __int64 v86; // [rsp+160h] [rbp-350h]
  __int64 v87; // [rsp+170h] [rbp-340h] BYREF
  __int64 *v88; // [rsp+178h] [rbp-338h]
  unsigned int v89; // [rsp+180h] [rbp-330h]
  unsigned int v90; // [rsp+184h] [rbp-32Ch]
  char v91; // [rsp+18Ch] [rbp-324h]
  _BYTE v92[64]; // [rsp+190h] [rbp-320h] BYREF
  unsigned __int64 v93; // [rsp+1D0h] [rbp-2E0h] BYREF
  unsigned __int64 v94; // [rsp+1D8h] [rbp-2D8h]
  unsigned __int64 v95; // [rsp+1E0h] [rbp-2D0h]
  char v96[8]; // [rsp+1F0h] [rbp-2C0h] BYREF
  unsigned __int64 v97; // [rsp+1F8h] [rbp-2B8h]
  char v98; // [rsp+20Ch] [rbp-2A4h]
  _BYTE v99[64]; // [rsp+210h] [rbp-2A0h] BYREF
  unsigned __int64 v100; // [rsp+250h] [rbp-260h]
  unsigned __int64 v101; // [rsp+258h] [rbp-258h]
  unsigned __int64 v102; // [rsp+260h] [rbp-250h]
  __m128i v103; // [rsp+270h] [rbp-240h] BYREF
  __int64 v104; // [rsp+280h] [rbp-230h] BYREF
  __int64 v105; // [rsp+288h] [rbp-228h]
  char v106[64]; // [rsp+290h] [rbp-220h] BYREF
  const __m128i *v107; // [rsp+2D0h] [rbp-1E0h]
  __int64 v108; // [rsp+2D8h] [rbp-1D8h]
  unsigned __int64 v109; // [rsp+2E0h] [rbp-1D0h]
  char v110[8]; // [rsp+2E8h] [rbp-1C8h] BYREF
  unsigned __int64 v111; // [rsp+2F0h] [rbp-1C0h]
  char v112; // [rsp+304h] [rbp-1ACh]
  char v113[64]; // [rsp+308h] [rbp-1A8h] BYREF
  const __m128i *v114; // [rsp+348h] [rbp-168h]
  const __m128i *v115; // [rsp+350h] [rbp-160h]
  unsigned __int64 v116; // [rsp+358h] [rbp-158h]

  v2 = *(_QWORD *)(a1 + 16);
  if ( *(_BYTE *)(v2 + 112) )
  {
    *(_DWORD *)(v2 + 116) = 0;
    v2 = *(_QWORD *)(a1 + 16);
  }
  else
  {
    v103.m128i_i32[3] = 32;
    v103.m128i_i64[0] = (__int64)&v104;
    v3 = *(_QWORD *)(v2 + 96);
    if ( v3 )
    {
      v4 = *(_QWORD *)(v3 + 24);
      v5 = &v104;
      v104 = *(_QWORD *)(v2 + 96);
      v6 = 1;
      v103.m128i_i32[2] = 1;
      v105 = v4;
      *(_DWORD *)(v3 + 72) = 0;
      v7 = 1;
      do
      {
        while ( 1 )
        {
          v72 = v6++;
          v13 = &v5[2 * v7 - 2];
          v14 = (__int64 *)v13[1];
          if ( v14 != (__int64 *)(*(_QWORD *)(*v13 + 24) + 8LL * *(unsigned int *)(*v13 + 32)) )
            break;
          --v7;
          *(_DWORD *)(*v13 + 76) = v72;
          v103.m128i_i32[2] = v7;
          if ( !v7 )
            goto LABEL_9;
        }
        v8 = *v14;
        v13[1] = (__int64)(v14 + 1);
        v9 = v103.m128i_u32[2];
        v10 = *(_QWORD *)(v8 + 24);
        v11 = v103.m128i_u32[2] + 1LL;
        if ( v11 > v103.m128i_u32[3] )
        {
          v73 = v72;
          v74 = *(_QWORD *)(v8 + 24);
          v75 = v8;
          sub_C8D5F0((__int64)&v103, &v104, v11, 0x10u, v8, v72);
          v5 = (__int64 *)v103.m128i_i64[0];
          v9 = v103.m128i_u32[2];
          LODWORD(v72) = v73;
          v10 = v74;
          v8 = v75;
        }
        v12 = &v5[2 * v9];
        *v12 = v8;
        v12[1] = v10;
        v7 = ++v103.m128i_i32[2];
        *(_DWORD *)(v8 + 72) = v72;
        v5 = (__int64 *)v103.m128i_i64[0];
      }
      while ( v7 );
LABEL_9:
      *(_DWORD *)(v2 + 116) = 0;
      *(_BYTE *)(v2 + 112) = 1;
      if ( v5 != &v104 )
        _libc_free((unsigned __int64)v5);
      v2 = *(_QWORD *)(a1 + 16);
    }
  }
  v15 = *(_QWORD *)(v2 + 96);
  v76[0] = (unsigned __int64)v77;
  v76[1] = 0x800000000LL;
  memset(v78, 0, 0x78u);
  v79.m128i_i64[1] = (__int64)v83;
  v78[1] = (unsigned __int64)&v78[4];
  v80 = 0x100000008LL;
  v83[0] = v15;
  v103.m128i_i64[0] = v15;
  LODWORD(v78[2]) = 8;
  BYTE4(v78[3]) = 1;
  v84 = 0;
  v85 = 0;
  v86 = 0;
  v81 = 0;
  v82 = 1;
  v79.m128i_i64[0] = 1;
  LOBYTE(v104) = 0;
  sub_2A45C60((__int64)&v84, &v103);
  sub_C8CF70((__int64)v96, v99, 8, (__int64)&v78[4], (__int64)v78);
  v16 = v78[12];
  memset(&v78[12], 0, 24);
  v100 = v16;
  v101 = v78[13];
  v102 = v78[14];
  sub_C8CF70((__int64)&v87, v92, 8, (__int64)v83, (__int64)&v79);
  v17 = v84;
  v84 = 0;
  v93 = v17;
  v18 = v85;
  v85 = 0;
  v94 = v18;
  v19 = v86;
  v86 = 0;
  v95 = v19;
  sub_C8CF70((__int64)&v103, v106, 8, (__int64)v92, (__int64)&v87);
  v20 = v93;
  v93 = 0;
  v107 = (const __m128i *)v20;
  v21 = v94;
  v94 = 0;
  v108 = v21;
  v22 = v95;
  v95 = 0;
  v109 = v22;
  sub_C8CF70((__int64)v110, v113, 8, (__int64)v99, (__int64)v96);
  v26 = v100;
  v100 = 0;
  v114 = (const __m128i *)v26;
  v27 = v101;
  v101 = 0;
  v115 = (const __m128i *)v27;
  v28 = v102;
  v102 = 0;
  v116 = v28;
  if ( v93 )
    j_j___libc_free_0(v93);
  if ( !v91 )
    _libc_free((unsigned __int64)v88);
  if ( v100 )
    j_j___libc_free_0(v100);
  if ( !v98 )
    _libc_free(v97);
  if ( v84 )
    j_j___libc_free_0(v84);
  if ( !v82 )
    _libc_free(v79.m128i_u64[1]);
  if ( v78[12] )
    j_j___libc_free_0(v78[12]);
  if ( !BYTE4(v78[3]) )
    _libc_free(v78[1]);
  v29 = (const __m128i *)v92;
  sub_C8CD80((__int64)&v87, (__int64)v92, (__int64)&v103, v23, v24, v25);
  v33 = v108;
  v34 = v107;
  v93 = 0;
  v94 = 0;
  v95 = 0;
  v35 = v108 - (_QWORD)v107;
  if ( (const __m128i *)v108 == v107 )
  {
    v37 = 0;
  }
  else
  {
    if ( v35 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_118;
    v36 = sub_22077B0(v108 - (_QWORD)v107);
    v33 = v108;
    v34 = v107;
    v37 = v36;
  }
  v93 = v37;
  v94 = v37;
  v95 = v37 + v35;
  if ( (const __m128i *)v33 != v34 )
  {
    v38 = (__m128i *)v37;
    v39 = v34;
    do
    {
      if ( v38 )
      {
        *v38 = _mm_loadu_si128(v39);
        v31 = v39[1].m128i_i64[0];
        v38[1].m128i_i64[0] = v31;
      }
      v39 = (const __m128i *)((char *)v39 + 24);
      v38 = (__m128i *)((char *)v38 + 24);
    }
    while ( (const __m128i *)v33 != v39 );
    v37 += 8 * ((unsigned __int64)(v33 - 24 - (_QWORD)v34) >> 3) + 24;
  }
  v94 = v37;
  v34 = (const __m128i *)v96;
  sub_C8CD80((__int64)v96, (__int64)v99, (__int64)v110, v33, v31, v32);
  v42 = v115;
  v29 = v114;
  v100 = 0;
  v101 = 0;
  v102 = 0;
  v43 = (char *)v115 - (char *)v114;
  if ( v115 != v114 )
  {
    if ( v43 <= 0x7FFFFFFFFFFFFFF8LL )
    {
      v44 = sub_22077B0((char *)v115 - (char *)v114);
      v42 = v115;
      v29 = v114;
      v45 = v44;
      goto LABEL_40;
    }
LABEL_118:
    sub_4261EA(v34, v29, v30);
  }
  v45 = 0;
LABEL_40:
  v100 = v45;
  v46 = (__m128i *)v45;
  v101 = v45;
  v102 = v45 + v43;
  if ( v29 != v42 )
  {
    v47 = v29;
    do
    {
      if ( v46 )
      {
        *v46 = _mm_loadu_si128(v47);
        v40 = (__int64 *)v47[1].m128i_i64[0];
        v46[1].m128i_i64[0] = (__int64)v40;
      }
      v47 = (const __m128i *)((char *)v47 + 24);
      v46 = (__m128i *)((char *)v46 + 24);
    }
    while ( v47 != v42 );
    v46 = (__m128i *)(v45 + 8 * ((unsigned __int64)((char *)&v47[-2].m128i_u64[1] - (char *)v29) >> 3) + 24);
  }
  v48 = v94;
  v49 = v93;
  v101 = (unsigned __int64)v46;
  v50 = v76;
  v51 = (__int64)v46->m128i_i64 - v45;
  if ( v94 - v93 != v51 )
    goto LABEL_47;
LABEL_66:
  if ( v48 != v49 )
  {
    v51 = v45;
    while ( 1 )
    {
      v50 = *(unsigned __int64 **)v51;
      if ( *(_QWORD *)v49 != *(_QWORD *)v51 )
        break;
      v61 = *(_BYTE *)(v49 + 16);
      if ( v61 != *(_BYTE *)(v51 + 16) )
        break;
      if ( v61 )
      {
        v50 = *(unsigned __int64 **)(v51 + 8);
        if ( *(unsigned __int64 **)(v49 + 8) != v50 )
          break;
      }
      v49 += 24LL;
      v51 += 24;
      if ( v48 == v49 )
        goto LABEL_73;
    }
    while ( 1 )
    {
LABEL_47:
      v52 = *(_QWORD *)(v48 - 24);
      v53 = *(_QWORD *)v52;
      v54 = *(_QWORD *)(*(_QWORD *)v52 + 48LL) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v54 == *(_QWORD *)v52 + 48LL
        || !v54
        || (v41 = (__int64 *)(v54 - 24), (unsigned int)*(unsigned __int8 *)(v54 - 24) - 30 > 0xA) )
      {
        BUG();
      }
      v55 = *(_BYTE *)(v54 - 24);
      if ( v55 == 31 )
      {
        if ( (*(_DWORD *)(v54 - 20) & 0x7FFFFFF) == 3 )
        {
          v50 = *(unsigned __int64 **)(v54 - 56);
          if ( *(unsigned __int64 **)(v54 - 88) != v50 )
          {
            sub_2A486D0(a1, v54 - 24, v53, (__int64)v76, (__int64)v40, (__int64)v41);
            v48 = v94;
            v52 = *(_QWORD *)(v94 - 24);
          }
        }
      }
      else if ( v55 == 32 )
      {
        sub_2A48F20(a1, v54 - 24, v53, (__int64)v76);
        v48 = v94;
LABEL_111:
        v52 = *(_QWORD *)(v48 - 24);
      }
      if ( !*(_BYTE *)(v48 - 8) )
      {
        v56 = *(__int64 **)(v52 + 24);
        *(_BYTE *)(v48 - 8) = 1;
        *(_QWORD *)(v48 - 16) = v56;
        goto LABEL_56;
      }
      while ( 1 )
      {
        v56 = *(__int64 **)(v48 - 16);
LABEL_56:
        if ( v56 == (__int64 *)(*(_QWORD *)(v52 + 24) + 8LL * *(unsigned int *)(v52 + 32)) )
          break;
        v57 = v56 + 1;
        *(_QWORD *)(v48 - 16) = v56 + 1;
        v58 = *v56;
        if ( !v91 )
          goto LABEL_63;
        v59 = v88;
        v57 = &v88[v90];
        if ( v88 == v57 )
        {
LABEL_108:
          if ( v90 < v89 )
          {
            ++v90;
            *v57 = v58;
            ++v87;
LABEL_64:
            v79.m128i_i64[0] = v58;
            LOBYTE(v80) = 0;
            sub_2A45C60((__int64)&v93, &v79);
            v49 = v93;
            v48 = v94;
            goto LABEL_65;
          }
LABEL_63:
          sub_C8CC70((__int64)&v87, v58, (__int64)v57, (__int64)v50, (__int64)v40, (__int64)v41);
          if ( v60 )
            goto LABEL_64;
        }
        else
        {
          while ( v58 != *v59 )
          {
            if ( v57 == ++v59 )
              goto LABEL_108;
          }
        }
      }
      v94 -= 24LL;
      v49 = v93;
      v48 = v94;
      if ( v94 != v93 )
        goto LABEL_111;
LABEL_65:
      v45 = v100;
      v51 = v101 - v100;
      if ( v48 - v49 == v101 - v100 )
        goto LABEL_66;
    }
  }
LABEL_73:
  v62 = v102 - v45;
  if ( v45 )
    j_j___libc_free_0(v45);
  if ( !v98 )
    _libc_free(v97);
  if ( v93 )
  {
    v62 = v95 - v93;
    j_j___libc_free_0(v93);
  }
  if ( !v91 )
    _libc_free((unsigned __int64)v88);
  if ( v114 )
  {
    v62 = v116 - (_QWORD)v114;
    j_j___libc_free_0((unsigned __int64)v114);
  }
  if ( !v112 )
    _libc_free(v111);
  if ( v107 )
  {
    v62 = v109 - (_QWORD)v107;
    j_j___libc_free_0((unsigned __int64)v107);
  }
  if ( !BYTE4(v105) )
    _libc_free(v103.m128i_u64[1]);
  v63 = *(_QWORD *)(a1 + 24);
  if ( !*(_BYTE *)(v63 + 192) )
    sub_CFDFC0(*(_QWORD *)(a1 + 24), v62, v51, (__int64)v50, (__int64)v40, v41);
  v64 = *(_QWORD *)(v63 + 16);
  v65 = v64 + 32LL * *(unsigned int *)(v63 + 24);
  while ( v64 != v65 )
  {
    while ( 1 )
    {
      v66 = *(_QWORD *)(v64 + 16);
      if ( v66 )
      {
        if ( *(_BYTE *)v66 == 85 )
        {
          v67 = *(_QWORD *)(v66 - 32);
          if ( v67 )
          {
            if ( !*(_BYTE *)v67 && *(_QWORD *)(v67 + 24) == *(_QWORD *)(v66 + 80) && (*(_BYTE *)(v67 + 33) & 0x20) != 0 )
            {
              v68 = *(_QWORD *)(v66 + 40);
              v69 = *(_QWORD *)(a1 + 16);
              if ( v68 )
              {
                v70 = (unsigned int)(*(_DWORD *)(v68 + 44) + 1);
                v71 = *(_DWORD *)(v68 + 44) + 1;
              }
              else
              {
                v70 = 0;
                v71 = 0;
              }
              if ( v71 < *(_DWORD *)(v69 + 32) && *(_QWORD *)(*(_QWORD *)(v69 + 24) + 8 * v70) )
                break;
            }
          }
        }
      }
      v64 += 32;
      if ( v64 == v65 )
        goto LABEL_105;
    }
    v64 += 32;
    sub_2A482C0(a1, v66, v68, (__int64)v76, v40, (__int64)v41);
  }
LABEL_105:
  sub_2A49D10(a1, (__int64)v76);
  if ( (_BYTE *)v76[0] != v77 )
    _libc_free(v76[0]);
}
