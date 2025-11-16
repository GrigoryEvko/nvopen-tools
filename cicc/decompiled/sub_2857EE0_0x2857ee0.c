// Function: sub_2857EE0
// Address: 0x2857ee0
//
__int64 __fastcall sub_2857EE0(__int64 a1, __int64 a2, __int64 *a3, unsigned int a4, _QWORD **a5, __int64 a6)
{
  unsigned int v6; // r15d
  __int64 v8; // rbx
  __m128i v9; // rax
  __int64 *v10; // rdi
  __int64 v11; // r12
  char v12; // r14
  char v13; // al
  __int64 v14; // r10
  unsigned int v15; // esi
  __int64 v16; // r9
  unsigned __int64 v17; // rax
  __int64 v18; // r8
  unsigned int v19; // edi
  _QWORD *v20; // r10
  __int64 v21; // rcx
  char v22; // al
  __int64 v23; // r8
  __int64 v24; // rax
  unsigned __int64 v25; // rcx
  __int64 v26; // r13
  bool v27; // cc
  unsigned __int64 v28; // rcx
  _QWORD *v29; // r15
  int v30; // eax
  __int64 v31; // r13
  __int64 v32; // rdx
  __int64 v33; // rax
  __m128i v34; // xmm1
  __int64 v35; // rdx
  __m128i v36; // xmm2
  __m128i v37; // xmm3
  __int64 v38; // rax
  __m128i v39; // xmm0
  __int64 v40; // rax
  __m128i v42; // xmm7
  __int64 v43; // rax
  __int64 v44; // r15
  __int64 v45; // r13
  __int64 v46; // rax
  unsigned __int64 v47; // rdi
  __int64 v48; // rbx
  __int64 v49; // r15
  __int64 v50; // rdx
  __int64 v51; // rcx
  __int64 v52; // rcx
  __int64 v53; // rdx
  __int64 v54; // rcx
  __m128i v55; // xmm4
  unsigned __int64 v56; // rbx
  __int64 v57; // rdx
  unsigned __int64 *v58; // rbx
  unsigned __int64 *v59; // r14
  int v60; // r13d
  unsigned __int64 v61; // r15
  unsigned __int64 v62; // r13
  unsigned __int64 v63; // rbx
  unsigned __int64 v64; // r14
  unsigned __int64 v65; // rdi
  unsigned __int64 v66; // r14
  int v67; // ecx
  int v68; // ecx
  int v69; // esi
  int v70; // esi
  unsigned int v71; // edx
  __int64 v72; // rdi
  int v73; // r11d
  int v74; // esi
  int v75; // esi
  int v76; // r11d
  unsigned int v77; // edx
  __int64 v78; // rdi
  __int64 v79; // [rsp-10h] [rbp-9A0h]
  __int64 v80; // [rsp-8h] [rbp-998h]
  __int64 v81; // [rsp+0h] [rbp-990h]
  __int64 v82; // [rsp+8h] [rbp-988h]
  __int64 v83; // [rsp+10h] [rbp-980h]
  char v84; // [rsp+10h] [rbp-980h]
  char v85; // [rsp+18h] [rbp-978h]
  __int64 v87; // [rsp+20h] [rbp-970h]
  __int64 v88; // [rsp+28h] [rbp-968h]
  _QWORD *v89; // [rsp+28h] [rbp-968h]
  __int64 v90; // [rsp+28h] [rbp-968h]
  int v91; // [rsp+28h] [rbp-968h]
  unsigned __int64 v93; // [rsp+30h] [rbp-960h]
  unsigned __int64 v94; // [rsp+30h] [rbp-960h]
  __int64 v97; // [rsp+40h] [rbp-950h]
  __int64 v98; // [rsp+40h] [rbp-950h]
  _QWORD *v99; // [rsp+40h] [rbp-950h]
  unsigned int v100; // [rsp+40h] [rbp-950h]
  unsigned __int64 v102; // [rsp+58h] [rbp-938h] BYREF
  __m128i v103; // [rsp+60h] [rbp-930h] BYREF
  _QWORD *v104; // [rsp+70h] [rbp-920h]
  __int64 v105; // [rsp+78h] [rbp-918h]
  _QWORD v106[6]; // [rsp+80h] [rbp-910h] BYREF
  __int64 v107; // [rsp+B0h] [rbp-8E0h] BYREF
  _QWORD v108[4]; // [rsp+D0h] [rbp-8C0h] BYREF
  unsigned int v109; // [rsp+F0h] [rbp-8A0h]
  _QWORD **v110; // [rsp+F8h] [rbp-898h]
  unsigned int v111; // [rsp+100h] [rbp-890h]
  char *v112; // [rsp+108h] [rbp-888h]
  __int64 v113; // [rsp+110h] [rbp-880h]
  char v114; // [rsp+118h] [rbp-878h] BYREF
  __int64 v115; // [rsp+398h] [rbp-5F8h]
  char v116; // [rsp+3A0h] [rbp-5F0h]
  unsigned __int64 v117; // [rsp+3A8h] [rbp-5E8h]
  char v118; // [rsp+3B0h] [rbp-5E0h]
  __int16 v119; // [rsp+3B8h] [rbp-5D8h]
  __int64 v120; // [rsp+3C0h] [rbp-5D0h]
  char *v121; // [rsp+3C8h] [rbp-5C8h]
  __int64 v122; // [rsp+3D0h] [rbp-5C0h]
  char v123; // [rsp+3D8h] [rbp-5B8h] BYREF
  __int64 v124; // [rsp+918h] [rbp-78h]
  char *v125; // [rsp+920h] [rbp-70h]
  __int64 v126; // [rsp+928h] [rbp-68h]
  int v127; // [rsp+930h] [rbp-60h]
  char v128; // [rsp+934h] [rbp-5Ch]
  char v129; // [rsp+938h] [rbp-58h] BYREF

  v6 = a6;
  v8 = a2;
  v88 = *a3;
  v9.m128i_i64[0] = sub_28579B0((__int64)a3, *(__int64 **)(a2 + 8));
  v10 = *(__int64 **)(a2 + 48);
  v11 = v9.m128i_i64[0];
  v12 = v9.m128i_i8[8];
  v103 = v9;
  v13 = sub_2850840(v10, a4, (__int64)a5, v6, v9.m128i_u64[0], v9.m128i_i8[8], 1u);
  v14 = v88;
  if ( v13 )
  {
    v14 = *a3;
  }
  else
  {
    *a3 = v88;
    v12 = 0;
    v11 = 0;
  }
  v15 = *(_DWORD *)(a2 + 37520);
  v16 = v8 + 37496;
  v17 = (2LL * a4) | v14 & 0xFFFFFFFFFFFFFFF9LL;
  if ( !v15 )
  {
    ++*(_QWORD *)(v8 + 37496);
    goto LABEL_72;
  }
  v18 = *(_QWORD *)(v8 + 37504);
  v19 = (v15 - 1) & (v17 ^ (v17 >> 9));
  v20 = (_QWORD *)(v18 + 16LL * v19);
  v21 = *v20;
  if ( v17 != *v20 )
  {
    v91 = 1;
    v99 = 0;
    while ( v21 != -2 )
    {
      if ( v21 == -16 )
      {
        if ( v99 )
          v20 = v99;
        v99 = v20;
      }
      v19 = (v15 - 1) & (v91 + v19);
      v20 = (_QWORD *)(v18 + 16LL * v19);
      v21 = *v20;
      if ( v17 == *v20 )
        goto LABEL_5;
      ++v91;
    }
    if ( v99 )
      v20 = v99;
    v67 = *(_DWORD *)(v8 + 37512);
    ++*(_QWORD *)(v8 + 37496);
    v68 = v67 + 1;
    if ( 4 * v68 < 3 * v15 )
    {
      v23 = v15 >> 3;
      if ( v15 - *(_DWORD *)(v8 + 37516) - v68 > (unsigned int)v23 )
      {
LABEL_66:
        *(_DWORD *)(v8 + 37512) = v68;
        if ( *v20 != -2 )
          --*(_DWORD *)(v8 + 37516);
        *v20 = v17;
        v20[1] = 0;
        goto LABEL_7;
      }
      v100 = v17 ^ (v17 >> 9);
      v94 = v17;
      sub_2854A40(v8 + 37496, v15);
      v74 = *(_DWORD *)(v8 + 37520);
      if ( v74 )
      {
        v75 = v74 - 1;
        v16 = 0;
        v23 = *(_QWORD *)(v8 + 37504);
        v76 = 1;
        v77 = v75 & v100;
        v68 = *(_DWORD *)(v8 + 37512) + 1;
        v17 = v94;
        v20 = (_QWORD *)(v23 + 16LL * (v75 & v100));
        v78 = *v20;
        if ( v94 == *v20 )
          goto LABEL_66;
        while ( v78 != -2 )
        {
          if ( !v16 && v78 == -16 )
            v16 = (__int64)v20;
          v77 = v75 & (v76 + v77);
          v20 = (_QWORD *)(v23 + 16LL * v77);
          v78 = *v20;
          if ( v94 == *v20 )
            goto LABEL_66;
          ++v76;
        }
        goto LABEL_76;
      }
      goto LABEL_98;
    }
LABEL_72:
    v93 = v17;
    sub_2854A40(v8 + 37496, 2 * v15);
    v69 = *(_DWORD *)(v8 + 37520);
    if ( v69 )
    {
      v17 = v93;
      v70 = v69 - 1;
      v23 = *(_QWORD *)(v8 + 37504);
      v68 = *(_DWORD *)(v8 + 37512) + 1;
      v71 = v70 & (v93 ^ (v93 >> 9));
      v20 = (_QWORD *)(v23 + 16LL * v71);
      v72 = *v20;
      if ( v93 == *v20 )
        goto LABEL_66;
      v73 = 1;
      v16 = 0;
      while ( v72 != -2 )
      {
        if ( !v16 && v72 == -16 )
          v16 = (__int64)v20;
        v71 = v70 & (v73 + v71);
        v20 = (_QWORD *)(v23 + 16LL * v71);
        v72 = *v20;
        if ( v93 == *v20 )
          goto LABEL_66;
        ++v73;
      }
LABEL_76:
      if ( v16 )
        v20 = (_QWORD *)v16;
      goto LABEL_66;
    }
LABEL_98:
    ++*(_DWORD *)(v8 + 37512);
    BUG();
  }
LABEL_5:
  v89 = v20;
  v97 = v20[1];
  v22 = sub_2850900(v8, *(_QWORD *)(v8 + 1320) + 2184 * v97, v11, v12, 1u, a4, a5, a6);
  v23 = v79;
  v16 = v80;
  if ( v22 )
  {
    v103.m128i_i64[0] = v11;
    v103.m128i_i8[8] = v12;
    v42 = _mm_loadu_si128(&v103);
    *(_QWORD *)a1 = v97;
    *(__m128i *)(a1 + 8) = v42;
    return a1;
  }
  v20 = v89;
LABEL_7:
  v24 = *(unsigned int *)(v8 + 1328);
  v108[0] = 0;
  v20[1] = v24;
  v25 = *(unsigned int *)(v8 + 1332);
  v90 = v24;
  v109 = a4;
  v26 = *(unsigned int *)(v8 + 1328);
  v110 = a5;
  v112 = &v114;
  v113 = 0x800000000LL;
  v115 = 0x7FFFFFFFFFFFFFFFLL;
  v27 = v26 + 1 <= v25;
  v28 = *(_QWORD *)(v8 + 1320);
  v117 = 0x8000000000000000LL;
  v121 = &v123;
  v111 = v6;
  v29 = v108;
  v122 = 0xC00000000LL;
  v125 = &v129;
  v30 = v26;
  memset(&v108[1], 0, 24);
  v116 = 0;
  v118 = 0;
  v119 = 1;
  v120 = 0;
  v124 = 0;
  v126 = 4;
  v127 = 0;
  v128 = 1;
  if ( v27 )
  {
    v98 = v28;
  }
  else
  {
    if ( v28 > (unsigned __int64)v108 || (unsigned __int64)v108 >= v28 + 2184 * v26 )
    {
      v81 = -1;
      v85 = 0;
    }
    else
    {
      v85 = 1;
      v81 = 0xFF0FF0FF0FF0FF1LL * ((__int64)((__int64)v108 - v28) >> 3);
    }
    v87 = v8 + 1336;
    v43 = sub_C8D7D0(v8 + 1320, v8 + 1336, v26 + 1, 0x888u, &v102, v16);
    v44 = *(_QWORD *)(v8 + 1320);
    v98 = v43;
    v45 = v43;
    v46 = 2184LL * *(unsigned int *)(v8 + 1328);
    v47 = v44 + v46;
    if ( v44 != v44 + v46 )
    {
      v83 = v8;
      v48 = *(_QWORD *)(v8 + 1320);
      v49 = v44 + v46;
      do
      {
        if ( v45 )
        {
          *(_DWORD *)(v45 + 24) = 0;
          *(_QWORD *)(v45 + 8) = 0;
          *(_DWORD *)(v45 + 16) = 0;
          *(_DWORD *)(v45 + 20) = 0;
          *(_QWORD *)v45 = 1;
          v52 = *(_QWORD *)(v48 + 8);
          ++*(_QWORD *)v48;
          v53 = *(_QWORD *)(v45 + 8);
          *(_QWORD *)(v45 + 8) = v52;
          LODWORD(v52) = *(_DWORD *)(v48 + 16);
          *(_QWORD *)(v48 + 8) = v53;
          LODWORD(v53) = *(_DWORD *)(v45 + 16);
          *(_DWORD *)(v45 + 16) = v52;
          LODWORD(v52) = *(_DWORD *)(v48 + 20);
          *(_DWORD *)(v48 + 16) = v53;
          LODWORD(v53) = *(_DWORD *)(v45 + 20);
          *(_DWORD *)(v45 + 20) = v52;
          v54 = *(unsigned int *)(v48 + 24);
          *(_DWORD *)(v48 + 20) = v53;
          LODWORD(v53) = *(_DWORD *)(v45 + 24);
          *(_DWORD *)(v45 + 24) = v54;
          *(_DWORD *)(v48 + 24) = v53;
          *(_DWORD *)(v45 + 32) = *(_DWORD *)(v48 + 32);
          v55 = _mm_loadu_si128((const __m128i *)(v48 + 40));
          *(_QWORD *)(v45 + 56) = v45 + 72;
          *(_DWORD *)(v45 + 64) = 0;
          *(_DWORD *)(v45 + 68) = 8;
          *(__m128i *)(v45 + 40) = v55;
          if ( *(_DWORD *)(v48 + 64) )
            sub_2852510(v45 + 56, v48 + 56, v45 + 72, v54, v23);
          *(__m128i *)(v45 + 712) = _mm_loadu_si128((const __m128i *)(v48 + 712));
          *(__m128i *)(v45 + 728) = _mm_loadu_si128((const __m128i *)(v48 + 728));
          *(_BYTE *)(v45 + 744) = *(_BYTE *)(v48 + 744);
          *(_BYTE *)(v45 + 745) = *(_BYTE *)(v48 + 745);
          v50 = *(_QWORD *)(v48 + 752);
          *(_DWORD *)(v45 + 768) = 0;
          *(_QWORD *)(v45 + 752) = v50;
          *(_QWORD *)(v45 + 760) = v45 + 776;
          *(_DWORD *)(v45 + 772) = 12;
          v51 = *(unsigned int *)(v48 + 768);
          if ( (_DWORD)v51 )
            sub_28515F0(v45 + 760, v48 + 760, v45 + 776, v51, v23, v16);
          sub_C8CF70(v45 + 2120, (void *)(v45 + 2152), 4, v48 + 2152, v48 + 2120);
        }
        v48 += 2184;
        v45 += 2184;
      }
      while ( v49 != v48 );
      v8 = v83;
      v61 = *(_QWORD *)(v83 + 1320);
      v47 = v61;
      v62 = v61 + 2184LL * *(unsigned int *)(v83 + 1328);
      if ( v61 != v62 )
      {
        v84 = v12;
        v82 = v8;
        do
        {
          v62 -= 2184LL;
          if ( !*(_BYTE *)(v62 + 2148) )
            _libc_free(*(_QWORD *)(v62 + 2128));
          v63 = *(_QWORD *)(v62 + 760);
          v64 = v63 + 112LL * *(unsigned int *)(v62 + 768);
          if ( v63 != v64 )
          {
            do
            {
              v64 -= 112LL;
              v65 = *(_QWORD *)(v64 + 40);
              if ( v65 != v64 + 56 )
                _libc_free(v65);
            }
            while ( v63 != v64 );
            v63 = *(_QWORD *)(v62 + 760);
          }
          if ( v63 != v62 + 776 )
            _libc_free(v63);
          v56 = *(_QWORD *)(v62 + 56);
          v66 = v56 + 80LL * *(unsigned int *)(v62 + 64);
          if ( v56 != v66 )
          {
            do
            {
              v66 -= 80LL;
              if ( !*(_BYTE *)(v66 + 44) )
                _libc_free(*(_QWORD *)(v66 + 24));
            }
            while ( v56 != v66 );
            v56 = *(_QWORD *)(v62 + 56);
          }
          if ( v56 != v62 + 72 )
            _libc_free(v56);
          v57 = *(unsigned int *)(v62 + 24);
          if ( (_DWORD)v57 )
          {
            v58 = *(unsigned __int64 **)(v62 + 8);
            v104 = v106;
            v105 = 0x400000001LL;
            v59 = &v58[6 * v57];
            v106[0] = -1;
            v106[4] = &v107;
            v106[5] = 0x400000001LL;
            v107 = -2;
            do
            {
              if ( (unsigned __int64 *)*v58 != v58 + 2 )
                _libc_free(*v58);
              v58 += 6;
            }
            while ( v59 != v58 );
          }
          sub_C7D6A0(*(_QWORD *)(v62 + 8), 48LL * *(unsigned int *)(v62 + 24), 8);
        }
        while ( v61 != v62 );
        v8 = v82;
        v12 = v84;
        v47 = *(_QWORD *)(v82 + 1320);
      }
    }
    v60 = v102;
    if ( v87 != v47 )
      _libc_free(v47);
    *(_DWORD *)(v8 + 1332) = v60;
    v28 = v98;
    v26 = *(unsigned int *)(v8 + 1328);
    *(_QWORD *)(v8 + 1320) = v98;
    v29 = v108;
    v30 = v26;
    if ( v85 )
      v29 = (_QWORD *)(v98 + 2184 * v81);
  }
  v31 = v98 + 2184 * v26;
  if ( v31 )
  {
    *(_QWORD *)(v31 + 16) = 0;
    *(_QWORD *)(v31 + 8) = 0;
    *(_DWORD *)(v31 + 24) = 0;
    *(_QWORD *)v31 = 1;
    v32 = v29[1];
    ++*v29;
    v33 = *(_QWORD *)(v31 + 8);
    *(_QWORD *)(v31 + 8) = v32;
    LODWORD(v32) = *((_DWORD *)v29 + 4);
    v29[1] = v33;
    LODWORD(v33) = *(_DWORD *)(v31 + 16);
    *(_DWORD *)(v31 + 16) = v32;
    LODWORD(v32) = *((_DWORD *)v29 + 5);
    *((_DWORD *)v29 + 4) = v33;
    LODWORD(v33) = *(_DWORD *)(v31 + 20);
    *(_DWORD *)(v31 + 20) = v32;
    LODWORD(v32) = *((_DWORD *)v29 + 6);
    *((_DWORD *)v29 + 5) = v33;
    LODWORD(v33) = *(_DWORD *)(v31 + 24);
    *(_DWORD *)(v31 + 24) = v32;
    v34 = _mm_loadu_si128((const __m128i *)(v29 + 5));
    *((_DWORD *)v29 + 6) = v33;
    LODWORD(v33) = *((_DWORD *)v29 + 8);
    *(__m128i *)(v31 + 40) = v34;
    *(_DWORD *)(v31 + 32) = v33;
    *(_QWORD *)(v31 + 56) = v31 + 72;
    *(_QWORD *)(v31 + 64) = 0x800000000LL;
    v35 = *((unsigned int *)v29 + 16);
    if ( (_DWORD)v35 )
      sub_2852510(v31 + 56, (__int64)(v29 + 7), v35, v28, v23);
    v36 = _mm_loadu_si128((const __m128i *)(v29 + 89));
    v37 = _mm_loadu_si128((const __m128i *)(v29 + 91));
    *(_WORD *)(v31 + 744) = *((_WORD *)v29 + 372);
    v38 = v29[94];
    *(__m128i *)(v31 + 712) = v36;
    *(_QWORD *)(v31 + 752) = v38;
    *(_QWORD *)(v31 + 760) = v31 + 776;
    *(_QWORD *)(v31 + 768) = 0xC00000000LL;
    *(__m128i *)(v31 + 728) = v37;
    if ( *((_DWORD *)v29 + 192) )
      sub_28515F0(v31 + 760, (__int64)(v29 + 95), v35, v28, v23, v16);
    sub_C8CF70(v31 + 2120, (void *)(v31 + 2152), 4, (__int64)(v29 + 269), (__int64)(v29 + 265));
    v30 = *(_DWORD *)(v8 + 1328);
  }
  *(_DWORD *)(v8 + 1328) = v30 + 1;
  sub_2855330((__int64)v108);
  v103.m128i_i64[0] = v11;
  v103.m128i_i8[8] = v12;
  v39 = _mm_loadu_si128(&v103);
  v40 = *(_QWORD *)(v8 + 1320) + 2184 * v90;
  *(_QWORD *)(v40 + 712) = v11;
  *(_BYTE *)(v40 + 720) = v12;
  *(_QWORD *)(v40 + 728) = v11;
  *(_BYTE *)(v40 + 736) = v12;
  *(_QWORD *)a1 = v90;
  *(__m128i *)(a1 + 8) = v39;
  return a1;
}
