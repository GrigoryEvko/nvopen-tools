// Function: sub_304C610
// Address: 0x304c610
//
void __fastcall sub_304C610(__int64 a1, __int64 a2, __int64 a3, __int64 *a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r14
  __int64 v7; // r11
  __int64 v8; // r13
  int v10; // eax
  __int64 v11; // rdx
  _QWORD *v12; // rax
  __int64 v13; // rax
  char v14; // cl
  __int64 v15; // rsi
  __int32 v16; // eax
  __int64 *v17; // r14
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // r14
  __int64 v21; // r15
  __m128i v22; // rax
  __int128 v23; // rax
  int v24; // r9d
  __int128 v25; // rax
  int v26; // r9d
  __int128 v27; // rax
  __m128i v28; // xmm4
  int v29; // r9d
  __int64 v30; // rax
  __int64 v31; // r9
  __int64 v32; // rdx
  __int64 v33; // r15
  __int64 v34; // rdx
  __int64 v35; // r14
  __int64 *v36; // rdx
  __int64 v37; // rsi
  __int64 v38; // rsi
  int v39; // esi
  int v40; // edx
  __m128i v41; // rax
  __int64 v42; // r9
  __int32 v43; // r10d
  __int64 v44; // r11
  __int64 v45; // rax
  __int64 v46; // r8
  _QWORD *v47; // rax
  int v48; // esi
  _QWORD *v49; // rax
  __m128i v50; // xmm3
  const __m128i *v51; // rax
  bool v52; // zf
  __m128i v53; // xmm2
  __int64 v54; // r8
  __int64 v55; // r9
  __int64 v56; // r14
  __int64 v57; // rax
  __int64 v58; // rdx
  __int64 v59; // r15
  __int64 v60; // r11
  __int32 v61; // r10d
  unsigned __int64 v62; // rdx
  __int64 *v63; // rax
  __int32 v64; // edi
  unsigned __int32 v65; // eax
  unsigned __int32 v66; // r13d
  __int64 v67; // rax
  __m128i v68; // xmm0
  unsigned __int64 v69; // rdx
  unsigned int *v70; // rdx
  __int64 v71; // rax
  __int64 v72; // rdx
  __int64 v73; // r15
  __int64 v74; // r14
  __int128 v75; // rax
  int v76; // r9d
  __m128i v77; // rax
  __int128 v78; // rax
  int v79; // r9d
  __int64 v80; // rax
  __int64 v81; // r14
  __int64 v82; // rax
  __int64 v83; // rdx
  __int64 v84; // r15
  unsigned __int64 v85; // rdx
  __int64 v86; // rax
  __int64 *v87; // rax
  __int64 v88; // r10
  __int64 v89; // r15
  __int64 v90; // r9
  __int64 v91; // r14
  int v92; // eax
  int v93; // edx
  __int64 v94; // rax
  __int64 v95; // r8
  __int64 v96; // r9
  __int64 v97; // r14
  __int64 v98; // rax
  __int64 v99; // rdx
  __int64 v100; // r15
  __int64 *v101; // rax
  unsigned __int64 v102; // rcx
  __int64 v103; // rax
  __int64 *v104; // rax
  _OWORD *v105; // rdi
  __int128 v106; // [rsp-20h] [rbp-1B0h]
  __int128 v107; // [rsp-10h] [rbp-1A0h]
  __int128 v108; // [rsp-10h] [rbp-1A0h]
  __int128 v109; // [rsp-10h] [rbp-1A0h]
  __int128 v110; // [rsp+0h] [rbp-190h]
  __int128 v111; // [rsp+0h] [rbp-190h]
  __int64 v112; // [rsp+8h] [rbp-188h]
  __int64 v113; // [rsp+18h] [rbp-178h]
  __int64 v114; // [rsp+38h] [rbp-158h]
  __int64 v115; // [rsp+38h] [rbp-158h]
  __int64 v116; // [rsp+38h] [rbp-158h]
  __int64 v117; // [rsp+38h] [rbp-158h]
  __m128i v118; // [rsp+40h] [rbp-150h] BYREF
  __m128i v119; // [rsp+50h] [rbp-140h] BYREF
  __int64 v120; // [rsp+60h] [rbp-130h] BYREF
  int v121; // [rsp+68h] [rbp-128h]
  _BYTE *v122; // [rsp+70h] [rbp-120h] BYREF
  __int64 v123; // [rsp+78h] [rbp-118h]
  _BYTE v124[80]; // [rsp+80h] [rbp-110h] BYREF
  __m128i v125; // [rsp+D0h] [rbp-C0h] BYREF
  _OWORD v126[11]; // [rsp+E0h] [rbp-B0h] BYREF

  v7 = a2;
  v8 = a3;
  v10 = *(_DWORD *)(a2 + 24);
  if ( v10 == 234 )
  {
    if ( **(_WORD **)(a2 + 48) == 35 )
    {
      v15 = *(_QWORD *)(a2 + 80);
      v122 = (_BYTE *)v15;
      if ( v15 )
      {
        v118.m128i_i64[0] = v7;
        v119.m128i_i64[0] = (__int64)&v122;
        sub_B96E90((__int64)&v122, v15, 1);
        v16 = *(_DWORD *)(v118.m128i_i64[0] + 72);
        v17 = *(__int64 **)(v118.m128i_i64[0] + 40);
        v125.m128i_i64[0] = (__int64)v122;
        LODWORD(v123) = v16;
        if ( v122 )
        {
          sub_B96E90((__int64)&v125, (__int64)v122, 1);
          v16 = v123;
        }
      }
      else
      {
        v16 = *(_DWORD *)(v7 + 72);
        v17 = *(__int64 **)(v7 + 40);
        v125.m128i_i64[0] = 0;
        v119.m128i_i64[0] = (__int64)&v122;
        LODWORD(v123) = v16;
      }
      v125.m128i_i32[2] = v16;
      v18 = *v17;
      if ( **(_WORD **)(*v17 + 48) == 6 )
        v19 = v17[1];
      else
        v18 = sub_33FAF80((_DWORD)a4, 234, (unsigned int)&v125, 6, 0, a6, *(_OWORD *)v17);
      v20 = v18;
      v21 = v19;
      if ( v125.m128i_i64[0] )
        sub_B91220((__int64)&v125, v125.m128i_i64[0]);
      *((_QWORD *)&v110 + 1) = v21;
      *(_QWORD *)&v110 = v20;
      v22.m128i_i64[0] = sub_33FAF80((_DWORD)a4, 216, v119.m128i_i32[0], 5, 0, a6, v110);
      v118 = v22;
      *(_QWORD *)&v23 = sub_3400BD0((_DWORD)a4, 8, v119.m128i_i32[0], 6, 0, 0, 0);
      *((_QWORD *)&v107 + 1) = 2;
      v125.m128i_i64[1] = v21;
      *(_QWORD *)&v107 = &v125;
      v125.m128i_i64[0] = v20;
      v126[0] = v23;
      *(_QWORD *)&v25 = sub_33FC220((_DWORD)a4, 192, v119.m128i_i32[0], 6, 0, v24, v107);
      *(_QWORD *)&v27 = sub_33FAF80((_DWORD)a4, 216, v119.m128i_i32[0], 5, 0, v26, v25);
      v28 = _mm_load_si128(&v118);
      *((_QWORD *)&v108 + 1) = 2;
      *(_QWORD *)&v108 = &v125;
      v126[0] = v27;
      v125 = v28;
      v30 = sub_33FC220((_DWORD)a4, 156, v119.m128i_i32[0], 35, 0, v29, v108);
      v33 = v32;
      v34 = *(unsigned int *)(v8 + 8);
      v35 = v30;
      if ( v34 + 1 > (unsigned __int64)*(unsigned int *)(v8 + 12) )
      {
        sub_C8D5F0(v8, (const void *)(v8 + 16), v34 + 1, 0x10u, v34 + 1, v31);
        v34 = *(unsigned int *)(v8 + 8);
      }
      v36 = (__int64 *)(*(_QWORD *)v8 + 16 * v34);
      *v36 = v35;
      v36[1] = v33;
      v37 = (__int64)v122;
      ++*(_DWORD *)(v8 + 8);
      if ( v37 )
        sub_B91220(v119.m128i_i64[0], v37);
    }
  }
  else
  {
    if ( v10 <= 234 )
    {
      switch ( v10 )
      {
        case '/':
          sub_3048C30(a2, (__int64)a4, a3, (__int64)a4, a5, a6);
          return;
        case '2':
          sub_3030BD0(a2, (int)a4, a3);
          return;
        case '.':
          v11 = *(_QWORD *)(**(_QWORD **)(a2 + 40) + 96LL);
          v12 = *(_QWORD **)(v11 + 24);
          if ( *(_DWORD *)(v11 + 32) > 0x40u )
            v12 = (_QWORD *)*v12;
          if ( (((_DWORD)v12 - 9062) & 0xFFFFFFFB) == 0 || (_DWORD)v12 == 9144 )
            sub_3032470(a2, (__int64)a4, v8);
          return;
      }
LABEL_9:
      sub_C64ED0("Unhandled custom legalization", 1u);
    }
    if ( v10 == 317 )
    {
      sub_3030EB0(a2, a4, a3, (__int64)a4, a5, a6);
      return;
    }
    if ( v10 <= 317 )
    {
      if ( v10 != 298 )
        goto LABEL_9;
      v13 = *(_QWORD *)(a1 + 537016);
      v14 = 0;
      if ( *(_DWORD *)(v13 + 344) > 0x63u )
        v14 = *(_DWORD *)(v13 + 336) > 0x57u;
      sub_3035F40(a2, (__int64)a4, a3, v14);
    }
    else
    {
      if ( (v10 & 0xFFFFFFFD) != 0x154 )
        goto LABEL_9;
      if ( *(_DWORD *)(*(_QWORD *)(a4[5] + 16) + 344LL) <= 0x59u )
        sub_C64ED0("128b atomics not supported on this architecture!", 1u);
      v38 = *(_QWORD *)(a2 + 80);
      v120 = v38;
      if ( v38 )
      {
        v119.m128i_i64[0] = v7;
        sub_B96E90((__int64)&v120, v38, 1);
        v7 = v119.m128i_i64[0];
        v10 = *(_DWORD *)(v119.m128i_i64[0] + 24);
      }
      v39 = 8185;
      if ( v10 != 340 )
        v39 = 8213;
      v40 = *(_DWORD *)(v7 + 72);
      v118.m128i_i64[0] = v7;
      LOWORD(v6) = 8;
      v121 = v40;
      v41.m128i_i64[0] = sub_3400BD0((_DWORD)a4, v39, (unsigned int)&v120, 7, 0, 0, 0);
      v42 = v112;
      v43 = 2;
      v119 = v41;
      v44 = v118.m128i_i64[0];
      v122 = v124;
      v123 = 0x500000000LL;
      v45 = 0;
      v46 = 1;
      while ( 1 )
      {
        v47 = &v122[16 * v45];
        *v47 = v6;
        v47[1] = 0;
        v45 = (unsigned int)(v123 + 1);
        LODWORD(v123) = v123 + 1;
        if ( v43 == 1 )
          break;
        v46 = v45 + 1;
        v43 = 1;
        LOWORD(v6) = 8;
        if ( v45 + 1 > (unsigned __int64)HIDWORD(v123) )
        {
          v114 = v44;
          sub_C8D5F0((__int64)&v122, v124, v45 + 1, 0x10u, v46, v42);
          v45 = (unsigned int)v123;
          v44 = v114;
          v43 = 1;
        }
      }
      if ( v45 + 1 > (unsigned __int64)HIDWORD(v123) )
      {
        v117 = v44;
        sub_C8D5F0((__int64)&v122, v124, v45 + 1, 0x10u, v46, v42);
        v45 = (unsigned int)v123;
        v44 = v117;
        v43 = 1;
      }
      v48 = 14;
      v49 = &v122[16 * v45];
      v50 = _mm_load_si128(&v119);
      v118.m128i_i32[0] = v43;
      *v49 = 1;
      v49[1] = 0;
      v125.m128i_i64[0] = (__int64)v126;
      v125.m128i_i64[1] = 0x800000000LL;
      v51 = *(const __m128i **)(v44 + 40);
      LODWORD(v123) = v123 + 1;
      v52 = *(_DWORD *)(v44 + 24) == 340;
      v53 = _mm_loadu_si128(v51);
      v119.m128i_i64[0] = v44;
      if ( !v52 )
        v48 = 0;
      v126[0] = v53;
      v126[1] = v50;
      v125.m128i_i32[2] = 2;
      v56 = sub_3400BD0((_DWORD)a4, (v48 << 16) | 5u, (unsigned int)&v120, 7, 0, 0, 0);
      v57 = v125.m128i_u32[2];
      v59 = v58;
      v60 = v119.m128i_i64[0];
      v61 = v118.m128i_i32[0];
      v62 = v125.m128i_u32[2] + 1LL;
      if ( v62 > v125.m128i_u32[3] )
      {
        v118.m128i_i64[0] = v119.m128i_i64[0];
        v119.m128i_i32[0] = v61;
        sub_C8D5F0((__int64)&v125, v126, v62, 0x10u, v54, v55);
        v57 = v125.m128i_u32[2];
        v60 = v118.m128i_i64[0];
        v61 = v119.m128i_i32[0];
      }
      v63 = (__int64 *)(v125.m128i_i64[0] + 16 * v57);
      *v63 = v56;
      v63[1] = v59;
      v64 = *(_DWORD *)(v60 + 64);
      v65 = ++v125.m128i_i32[2];
      v119.m128i_i32[0] = v64;
      if ( v64 != 1 )
      {
        v113 = v8;
        v66 = v61;
        do
        {
          while ( 1 )
          {
            v70 = (unsigned int *)(*(_QWORD *)(v60 + 40) + 40LL * v66);
            if ( *(_WORD *)(*(_QWORD *)(*(_QWORD *)v70 + 48LL) + 16LL * v70[2]) != 9 )
              break;
            v115 = v60;
            v71 = sub_33FAF80((_DWORD)a4, 234, (unsigned int)&v120, 116, 0, v55, *(_OWORD *)v70);
            v73 = v72;
            v74 = v71;
            *(_QWORD *)&v75 = sub_3400D50(a4, 0, &v120, 0);
            *((_QWORD *)&v106 + 1) = v73;
            *(_QWORD *)&v106 = v74;
            v77.m128i_i64[0] = sub_3406EB0((_DWORD)a4, 158, (unsigned int)&v120, 8, 0, v76, v106, v75);
            v118 = v77;
            *(_QWORD *)&v78 = sub_3400D50(a4, 1, &v120, 0);
            *((_QWORD *)&v109 + 1) = v73;
            *(_QWORD *)&v109 = v74;
            v80 = sub_3406EB0((_DWORD)a4, 158, (unsigned int)&v120, 8, 0, v79, v109, v78);
            v60 = v115;
            v81 = v80;
            v82 = v125.m128i_u32[2];
            v84 = v83;
            v85 = v125.m128i_u32[2] + 1LL;
            if ( v85 > v125.m128i_u32[3] )
            {
              sub_C8D5F0((__int64)&v125, v126, v85, 0x10u, v54, v55);
              v82 = v125.m128i_u32[2];
              v60 = v115;
            }
            *(__m128i *)(v125.m128i_i64[0] + 16 * v82) = _mm_load_si128(&v118);
            ++v125.m128i_i32[2];
            v86 = v125.m128i_u32[2];
            if ( (unsigned __int64)v125.m128i_u32[2] + 1 > v125.m128i_u32[3] )
            {
              v118.m128i_i64[0] = v60;
              sub_C8D5F0((__int64)&v125, v126, v125.m128i_u32[2] + 1LL, 0x10u, v54, v55);
              v86 = v125.m128i_u32[2];
              v60 = v118.m128i_i64[0];
            }
            v87 = (__int64 *)(v125.m128i_i64[0] + 16 * v86);
            ++v66;
            *v87 = v81;
            v87[1] = v84;
            ++v125.m128i_i32[2];
            if ( v119.m128i_i32[0] == v66 )
              goto LABEL_60;
          }
          v67 = v125.m128i_u32[2];
          v68 = _mm_loadu_si128((const __m128i *)v70);
          v69 = v125.m128i_u32[2] + 1LL;
          if ( v69 > v125.m128i_u32[3] )
          {
            v116 = v60;
            v118 = v68;
            sub_C8D5F0((__int64)&v125, v126, v69, 0x10u, v54, v55);
            v67 = v125.m128i_u32[2];
            v60 = v116;
            v68 = _mm_load_si128(&v118);
          }
          ++v66;
          *(__m128i *)(v125.m128i_i64[0] + 16 * v67) = v68;
          ++v125.m128i_i32[2];
        }
        while ( v119.m128i_i32[0] != v66 );
LABEL_60:
        v8 = v113;
        v65 = v125.m128i_u32[2];
      }
      v88 = *(_QWORD *)(v60 + 104);
      v89 = v65;
      v90 = *(unsigned __int16 *)(v60 + 96);
      v119.m128i_i64[0] = *(_QWORD *)(v60 + 112);
      v91 = v125.m128i_i64[0];
      v118.m128i_i64[0] = v90;
      v118.m128i_i64[1] = v88;
      v92 = sub_33E5830(a4, v122);
      v119.m128i_i64[0] = sub_33EA9D0(
                            (_DWORD)a4,
                            47,
                            (unsigned int)&v120,
                            v92,
                            v93,
                            v119.m128i_i32[0],
                            v91,
                            v89,
                            v118.m128i_i64[0],
                            v118.m128i_i64[1]);
      *((_QWORD *)&v111 + 1) = 1;
      *(_QWORD *)&v111 = v119.m128i_i64[0];
      v94 = sub_3406EB0((_DWORD)a4, 54, (unsigned int)&v120, 9, 0, v119.m128i_i32[0], v119.m128i_u64[0], v111);
      v96 = v119.m128i_i64[0];
      v97 = v94;
      v98 = *(unsigned int *)(v8 + 8);
      v100 = v99;
      if ( v98 + 1 > (unsigned __int64)*(unsigned int *)(v8 + 12) )
      {
        sub_C8D5F0(v8, (const void *)(v8 + 16), v98 + 1, 0x10u, v95, v119.m128i_i64[0]);
        v98 = *(unsigned int *)(v8 + 8);
        v96 = v119.m128i_i64[0];
      }
      v101 = (__int64 *)(*(_QWORD *)v8 + 16 * v98);
      *v101 = v97;
      v101[1] = v100;
      v102 = *(unsigned int *)(v8 + 12);
      v103 = (unsigned int)(*(_DWORD *)(v8 + 8) + 1);
      *(_DWORD *)(v8 + 8) = v103;
      if ( v103 + 1 > v102 )
      {
        v119.m128i_i64[0] = v96;
        sub_C8D5F0(v8, (const void *)(v8 + 16), v103 + 1, 0x10u, v95, v96);
        v103 = *(unsigned int *)(v8 + 8);
        v96 = v119.m128i_i64[0];
      }
      v104 = (__int64 *)(*(_QWORD *)v8 + 16 * v103);
      *v104 = v96;
      v104[1] = 2;
      v105 = (_OWORD *)v125.m128i_i64[0];
      ++*(_DWORD *)(v8 + 8);
      if ( v105 != v126 )
        _libc_free((unsigned __int64)v105);
      if ( v122 != v124 )
        _libc_free((unsigned __int64)v122);
      if ( v120 )
        sub_B91220((__int64)&v120, v120);
    }
  }
}
