// Function: sub_37A96C0
// Address: 0x37a96c0
//
__m128i *__fastcall sub_37A96C0(__int64 a1, __int64 a2, int a3, __m128i a4)
{
  int v4; // r11d
  __int64 v5; // rax
  __int64 v6; // r15
  unsigned __int64 v7; // r12
  __int64 v8; // r14
  __int64 v9; // r13
  unsigned __int8 *v10; // rbx
  __int64 v11; // rax
  __int64 v12; // r10
  unsigned __int16 v13; // cx
  __int64 v14; // r11
  unsigned int v15; // edx
  unsigned int v16; // r12d
  __int64 v17; // roff
  __m128i v18; // xmm0
  unsigned __int16 v19; // bx
  __int16 v20; // bx
  unsigned __int64 v21; // rdx
  char v22; // r14
  __m128i v23; // xmm1
  const __m128i *v24; // rsi
  char v25; // r14
  __int64 v26; // rsi
  _QWORD *v27; // r15
  __int64 v28; // rdi
  __m128i *v29; // rax
  __int32 v30; // edx
  __m128i *v31; // r12
  __int64 v33; // rsi
  __int64 v34; // rax
  unsigned int v35; // edx
  __int64 v36; // rdx
  __int16 v37; // cx
  unsigned int v38; // eax
  __int64 v39; // rcx
  __int64 v40; // r8
  __int64 v41; // r13
  __int64 v42; // r10
  __int64 v43; // rdx
  __int16 v44; // r11d^2
  unsigned __int16 v45; // ax
  unsigned int v46; // ebx
  __int64 *v47; // r13
  int v48; // eax
  __int64 v49; // r8
  __int64 v50; // r10
  unsigned int v51; // r11d
  __int64 v52; // rsi
  unsigned __int8 *v53; // rax
  __int64 v54; // rcx
  __int64 v55; // r8
  __int64 v56; // r10
  unsigned int v57; // edx
  unsigned __int8 *v58; // r11
  unsigned __int16 *v59; // rdx
  int v60; // eax
  __int64 v61; // rdx
  unsigned __int16 v62; // ax
  unsigned int v63; // ebx
  __int64 *v64; // r13
  int v65; // eax
  __int64 v66; // r8
  unsigned __int8 *v67; // r11
  __int64 v68; // r10
  unsigned int v69; // ebx
  unsigned __int8 *v70; // rax
  __int64 v71; // r10
  __int64 v72; // r11
  unsigned int v73; // edx
  unsigned __int16 v74; // cx
  __int64 v75; // r13
  unsigned int v76; // r15d
  unsigned __int16 v77; // ax
  unsigned __int64 v78; // rdx
  __int64 v79; // rdx
  bool v80; // al
  __int64 v81; // rdx
  __int64 v82; // r8
  unsigned __int16 v83; // ax
  __int64 v84; // rdx
  __int64 v85; // rdx
  __int64 v86; // rdx
  __int64 v87; // rdx
  __int64 *v88; // [rsp+10h] [rbp-120h]
  __int16 v89; // [rsp+12h] [rbp-11Eh]
  unsigned __int16 v90; // [rsp+18h] [rbp-118h]
  __int64 v91; // [rsp+18h] [rbp-118h]
  __int64 v92; // [rsp+18h] [rbp-118h]
  __int64 v93; // [rsp+18h] [rbp-118h]
  __int64 v94; // [rsp+18h] [rbp-118h]
  __int64 v95; // [rsp+20h] [rbp-110h]
  unsigned __int8 *v96; // [rsp+20h] [rbp-110h]
  __int64 v97; // [rsp+20h] [rbp-110h]
  __int16 v98; // [rsp+22h] [rbp-10Eh]
  __int16 v99; // [rsp+22h] [rbp-10Eh]
  __int64 v100; // [rsp+28h] [rbp-108h]
  int v101; // [rsp+34h] [rbp-FCh]
  unsigned __int64 v102; // [rsp+40h] [rbp-F0h]
  unsigned int v103; // [rsp+40h] [rbp-F0h]
  __int64 v104; // [rsp+48h] [rbp-E8h]
  __int64 v105; // [rsp+50h] [rbp-E0h]
  unsigned int v106; // [rsp+58h] [rbp-D8h]
  unsigned __int16 v107; // [rsp+58h] [rbp-D8h]
  unsigned __int64 v108; // [rsp+60h] [rbp-D0h]
  __int64 v109; // [rsp+60h] [rbp-D0h]
  unsigned int v110; // [rsp+68h] [rbp-C8h]
  const __m128i *v111; // [rsp+68h] [rbp-C8h]
  unsigned __int8 *v112; // [rsp+68h] [rbp-C8h]
  unsigned __int8 *v113; // [rsp+68h] [rbp-C8h]
  __int64 v115; // [rsp+70h] [rbp-C0h]
  __int64 v116; // [rsp+70h] [rbp-C0h]
  unsigned __int16 v118; // [rsp+78h] [rbp-B8h]
  unsigned __int16 v119; // [rsp+80h] [rbp-B0h] BYREF
  __int64 v120; // [rsp+88h] [rbp-A8h]
  __int64 v121; // [rsp+90h] [rbp-A0h] BYREF
  __int64 v122; // [rsp+98h] [rbp-98h]
  __m128i v123; // [rsp+A0h] [rbp-90h] BYREF
  unsigned __int64 v124; // [rsp+B0h] [rbp-80h]
  unsigned __int64 v125; // [rsp+B8h] [rbp-78h]
  unsigned __int8 *v126; // [rsp+C0h] [rbp-70h]
  unsigned __int64 v127; // [rsp+C8h] [rbp-68h]
  __m128i v128; // [rsp+D0h] [rbp-60h]
  __int64 v129; // [rsp+E0h] [rbp-50h]
  unsigned __int64 v130; // [rsp+E8h] [rbp-48h]
  __int64 v131; // [rsp+F0h] [rbp-40h]
  int v132; // [rsp+F8h] [rbp-38h]

  v5 = *(_QWORD *)(a2 + 40);
  v6 = *(_QWORD *)(v5 + 80);
  v105 = *(_QWORD *)(v5 + 48);
  v108 = *(_QWORD *)(v5 + 40);
  v7 = *(_QWORD *)(v5 + 160);
  v106 = *(_DWORD *)(v5 + 48);
  v8 = *(_QWORD *)(v5 + 168);
  v100 = *(_QWORD *)(v5 + 200);
  v9 = *(unsigned int *)(v5 + 168);
  v104 = *(_QWORD *)(v5 + 88);
  v110 = *(_DWORD *)(v5 + 88);
  v101 = *(_DWORD *)(v5 + 208);
  v102 = *(_QWORD *)(a2 + 104);
  if ( a3 == 1 )
  {
    v33 = *(_QWORD *)(v5 + 40);
    v98 = HIWORD(v4);
    v34 = sub_379AB60(a1, v108, v105);
    v106 = v35;
    v36 = *(_QWORD *)(v34 + 48) + 16LL * v35;
    v108 = v34;
    v37 = *(_WORD *)v36;
    v123.m128i_i64[1] = *(_QWORD *)(v36 + 8);
    v123.m128i_i16[0] = v37;
    v38 = sub_3281500(&v123, v33);
    v41 = *(_QWORD *)(v7 + 48) + 16 * v9;
    v42 = a2;
    v103 = v38;
    v43 = *(_QWORD *)(v41 + 8);
    v44 = v98;
    v119 = *(_WORD *)v41;
    v120 = v43;
    if ( v119 )
    {
      v91 = 0;
      v45 = word_4456580[v119 - 1];
    }
    else
    {
      v45 = sub_3009970((__int64)&v119, v33, v43, v39, v40);
      v44 = v98;
      v42 = a2;
      v91 = v79;
    }
    v99 = v44;
    v115 = v42;
    v46 = v45;
    v47 = *(__int64 **)(*(_QWORD *)(a1 + 8) + 64LL);
    LOWORD(v48) = sub_2D43050(v45, v103);
    v49 = 0;
    v50 = v115;
    HIWORD(v51) = v99;
    if ( !(_WORD)v48 )
    {
      v48 = sub_3009400(v47, v46, v91, v103, 0);
      v50 = v115;
      HIWORD(v51) = HIWORD(v48);
      v49 = v86;
    }
    LOWORD(v51) = v48;
    v52 = v7;
    v116 = v50;
    v53 = sub_3790540(a1, v7, v8, v51, v49, 0, a4);
    v56 = v116;
    v16 = v57;
    v58 = v53;
    v59 = (unsigned __int16 *)(*(_QWORD *)(v6 + 48) + 16LL * v110);
    v60 = *v59;
    v61 = *((_QWORD *)v59 + 1);
    LOWORD(v121) = v60;
    v122 = v61;
    if ( (_WORD)v60 )
    {
      v92 = 0;
      v62 = word_4456580[v60 - 1];
    }
    else
    {
      v113 = v58;
      v62 = sub_3009970((__int64)&v121, v52, v61, v54, v55);
      v56 = v116;
      v58 = v113;
      v92 = v85;
    }
    v95 = v56;
    v63 = v62;
    v112 = v58;
    v64 = *(__int64 **)(*(_QWORD *)(a1 + 8) + 64LL);
    LOWORD(v65) = sub_2D43050(v62, v103);
    v66 = 0;
    v67 = v112;
    v68 = v95;
    if ( !(_WORD)v65 )
    {
      v65 = sub_3009400(v64, v63, v92, v103, 0);
      v68 = v95;
      v67 = v112;
      v89 = HIWORD(v65);
      v66 = v87;
    }
    HIWORD(v69) = v89;
    v93 = v68;
    LOWORD(v69) = v65;
    v96 = v67;
    v70 = sub_3790540(a1, v6, v104, v69, v66, 1, a4);
    v71 = v93;
    v72 = (__int64)v96;
    v110 = v73;
    v10 = v70;
    v75 = *(_QWORD *)(v93 + 104);
    v123.m128i_i16[0] = *(_WORD *)(v93 + 96);
    v74 = v123.m128i_i16[0];
    v123.m128i_i64[1] = v75;
    if ( v123.m128i_i16[0] )
    {
      if ( (unsigned __int16)(v123.m128i_i16[0] - 17) <= 0xD3u )
      {
        v75 = 0;
        v74 = word_4456580[v123.m128i_u16[0] - 1];
      }
    }
    else
    {
      v80 = sub_30070B0((__int64)&v123);
      v72 = (__int64)v96;
      v74 = 0;
      v71 = v93;
      if ( v80 )
      {
        v83 = sub_3009970((__int64)&v123, v6, v81, 0, v82);
        v71 = v93;
        v72 = (__int64)v96;
        v74 = v83;
        v75 = v84;
      }
    }
    v94 = v71;
    v76 = v74;
    v97 = v72;
    v88 = *(__int64 **)(*(_QWORD *)(a1 + 8) + 64LL);
    v77 = sub_2D43050(v74, v103);
    v78 = 0;
    v14 = v97;
    v12 = v94;
    if ( !v77 )
    {
      v77 = sub_3009400(v88, v76, v75, v103, 0);
      v12 = v94;
      v14 = v97;
    }
    v102 = v78;
    v13 = v77;
  }
  else
  {
    v90 = *(_WORD *)(a2 + 96);
    if ( a3 != 4 )
      BUG();
    v10 = *(unsigned __int8 **)(v5 + 80);
    v11 = sub_379AB60(a1, v7, v8);
    v12 = a2;
    v13 = v90;
    v14 = v11;
    v16 = v15;
  }
  v17 = *(_QWORD *)(v12 + 40);
  v18 = _mm_loadu_si128((const __m128i *)v17);
  v126 = v10;
  v19 = *(_WORD *)(v12 + 32);
  v123 = v18;
  v125 = v106 | v105 & 0xFFFFFFFF00000000LL;
  v124 = v108;
  v20 = (v19 >> 7) & 7;
  v21 = v8 & 0xFFFFFFFF00000000LL;
  v22 = *(_BYTE *)(v12 + 33);
  v127 = v110 | v104 & 0xFFFFFFFF00000000LL;
  v23 = _mm_loadu_si128((const __m128i *)(v17 + 120));
  v24 = *(const __m128i **)(v12 + 112);
  v129 = v14;
  v25 = (v22 & 4) != 0;
  v130 = v16 | v21;
  v111 = v24;
  v26 = *(_QWORD *)(v12 + 80);
  v131 = v100;
  v121 = v26;
  v132 = v101;
  v128 = v23;
  v27 = *(_QWORD **)(a1 + 8);
  v28 = (__int64)v27;
  if ( v26 )
  {
    v107 = v13;
    v109 = v12;
    sub_B96E90((__int64)&v121, v26, 1);
    v13 = v107;
    v12 = v109;
    v28 = *(_QWORD *)(a1 + 8);
  }
  v118 = v13;
  LODWORD(v122) = *(_DWORD *)(v12 + 72);
  v29 = sub_33ED250(v28, 1, 0);
  v31 = sub_33E7ED0(
          v27,
          (unsigned __int64)v29,
          v30,
          v118,
          v102,
          (__int64)&v121,
          (unsigned __int64 *)&v123,
          6,
          v111,
          v20,
          v25);
  if ( v121 )
    sub_B91220((__int64)&v121, v121);
  return v31;
}
