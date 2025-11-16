// Function: sub_31E51E0
// Address: 0x31e51e0
//
__int64 __fastcall sub_31E51E0(__int64 a1, _QWORD *a2)
{
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rax
  signed __int64 v7; // r14
  __int32 v8; // eax
  unsigned __int64 v9; // rdi
  __int32 v10; // eax
  __int64 v11; // rax
  __int64 v12; // rdi
  void (*v13)(); // rax
  __int64 v14; // rax
  _DWORD *v15; // r15
  unsigned __int64 v16; // rax
  __int64 v17; // rdx
  const char *v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rdx
  __m128i *v21; // rcx
  __int64 v22; // rdx
  unsigned __int64 v23; // rax
  __int64 v24; // rdx
  void (*v25)(); // rax
  __int64 v26; // rax
  __int64 v27; // r14
  __int64 v28; // r11
  __int64 v29; // r10
  __int64 v30; // rax
  __int64 *v31; // r15
  __int64 *v32; // r14
  __int64 v33; // rax
  __int64 v34; // rdi
  void (*v35)(); // rax
  __int64 v36; // rdi
  void (*v37)(); // rax
  void (*v38)(void); // rax
  __int64 v39; // rax
  _BYTE *v40; // rsi
  __int64 v41; // rdx
  int v42; // r15d
  __int64 v43; // rax
  int v44; // ecx
  __int64 v45; // rdx
  __m128i *p_si128; // rsi
  __int64 v47; // rdx
  __int64 v48; // rdi
  void (*v49)(); // rax
  void (*v50)(); // rax
  __int64 v51; // rax
  unsigned __int64 v52; // r14
  __int64 v53; // rcx
  unsigned int v54; // eax
  _QWORD *v55; // r15
  __int64 v56; // r14
  int v57; // eax
  __int64 v58; // rax
  __int64 *v59; // r14
  __int64 v60; // r9
  __int64 v61; // rax
  unsigned __int64 v62; // rdi
  __m128i *v63; // rcx
  unsigned __int64 v64; // rsi
  unsigned __int64 v65; // r8
  int v66; // edx
  _QWORD *v67; // rax
  __int64 v68; // r14
  __int64 v69; // rax
  __int64 v70; // rax
  __int64 v71; // r9
  __int64 v72; // r14
  __int64 v73; // rdx
  unsigned __int64 v74; // rdi
  __m128i *v75; // rcx
  unsigned __int64 v76; // rsi
  __int64 v77; // r8
  int v78; // eax
  _QWORD *v79; // rdx
  __int64 *v80; // r13
  __int64 *v81; // r14
  __int64 v82; // rdi
  __int64 *v83; // rax
  __int64 *v84; // r13
  __int64 *i; // rbx
  __int64 v86; // rdi
  _QWORD *v88; // rdi
  __int64 v89; // rsi
  __int64 v90; // rax
  __int64 v91; // r9
  __int64 v92; // r14
  __int64 v93; // rdx
  unsigned __int64 v94; // rdi
  __m128i *v95; // rcx
  unsigned __int64 v96; // rsi
  __int64 v97; // r8
  int v98; // eax
  _QWORD *v99; // rdx
  void (__fastcall *v100)(__int64, __int64); // r15
  __int64 v101; // rax
  __int64 v102; // rax
  __int64 v103; // rax
  int v104; // eax
  __int64 v105; // rax
  __int64 v106; // rax
  __int64 v107; // rax
  __int64 v108; // rdx
  __int64 v109; // rdi
  __int8 *v110; // r13
  __int64 v111; // rdi
  __int8 *v112; // r15
  __int64 v113; // rax
  __int64 v114; // r8
  __int64 v115; // r9
  __int64 v116; // r14
  __int64 v117; // rcx
  unsigned __int64 v118; // rdi
  __m128i *v119; // r15
  unsigned __int64 v120; // rdx
  unsigned __int64 v121; // rsi
  int v122; // eax
  _QWORD *v123; // rdx
  __int64 v124; // rdi
  __int8 *v125; // r15
  __int64 v126; // rdi
  __int8 *v127; // r15
  __int64 v128; // [rsp-8h] [rbp-C8h]
  __int64 v129; // [rsp+0h] [rbp-C0h]
  int v130; // [rsp+0h] [rbp-C0h]
  __m128i *v131; // [rsp+8h] [rbp-B8h]
  __int64 v132; // [rsp+8h] [rbp-B8h]
  int v133; // [rsp+8h] [rbp-B8h]
  _QWORD v134[2]; // [rsp+10h] [rbp-B0h] BYREF
  unsigned __int64 v135; // [rsp+20h] [rbp-A0h] BYREF
  __int64 v136; // [rsp+28h] [rbp-98h]
  __int16 v137; // [rsp+40h] [rbp-80h]
  __m128i si128; // [rsp+50h] [rbp-70h] BYREF
  _QWORD v139[2]; // [rsp+60h] [rbp-60h] BYREF
  char v140; // [rsp+70h] [rbp-50h]
  char v141; // [rsp+71h] [rbp-4Fh]

  v4 = sub_B82360(*(_QWORD *)(a1 + 8), (__int64)&unk_50208C0);
  if ( v4 && (v5 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v4 + 104LL))(v4, &unk_50208C0)) != 0 )
    v6 = v5 + 176;
  else
    v6 = 0;
  *(_QWORD *)(a1 + 240) = v6;
  *(_WORD *)(a1 + 780) = 0;
  v7 = sub_BA8DC0((__int64)a2, (__int64)"llvm.dbg.cu", 11);
  v8 = 0;
  if ( v7 )
    v8 = sub_B91A00(v7);
  si128.m128i_i32[2] = v8;
  si128.m128i_i64[0] = v7;
  sub_BA95A0((__int64)&si128);
  v135 = v7;
  LODWORD(v136) = 0;
  sub_BA95A0((__int64)&v135);
  v9 = *(_QWORD *)(a1 + 448);
  v10 = si128.m128i_i32[2];
  *(_QWORD *)(a1 + 448) = 0;
  *(_BYTE *)(a1 + 782) = (_DWORD)v136 != v10;
  if ( v9 )
    sub_31D8060(v9);
  v11 = sub_31DA6B0(a1);
  (*(void (__fastcall **)(__int64, _QWORD, _QWORD))(*(_QWORD *)v11 + 24LL))(
    v11,
    *(_QWORD *)(a1 + 216),
    *(_QWORD *)(a1 + 200));
  v12 = sub_31DA6B0(a1);
  v13 = *(void (**)())(*(_QWORD *)v12 + 56LL);
  if ( v13 != nullsub_1713 )
    ((void (__fastcall *)(__int64, _QWORD *))v13)(v12, a2);
  v14 = *(_QWORD *)(a1 + 200);
  if ( *(_DWORD *)(v14 + 564) != 8 )
  {
    (*(void (__fastcall **)(_QWORD, _QWORD, _QWORD))(**(_QWORD **)(a1 + 224) + 192LL))(
      *(_QWORD *)(a1 + 224),
      0,
      *(_QWORD *)(v14 + 680));
    v15 = *(_DWORD **)(a1 + 200);
    if ( v15[141] == 5 )
    {
      v16 = (unsigned int)v15[139];
      if ( (unsigned int)v16 <= 0x1F )
      {
        v17 = 3623879202LL;
        if ( _bittest64(&v17, v16) )
        {
          v18 = sub_BAABC0((__int64)a2);
          v137 = 261;
          v135 = (unsigned __int64)v18;
          v136 = v19;
          sub_CC9F70((__int64)&si128, (void **)&v135);
          v129 = *(_QWORD *)(a1 + 224);
          v135 = sub_BAAC00((__int64)a2);
          v136 = v20;
          sub_BAABC0((__int64)a2);
          v21 = 0;
          if ( v22 )
            v21 = &si128;
          v131 = v21;
          v23 = sub_BAA900((__int64)a2);
          v134[1] = v24;
          v134[0] = v23;
          sub_E9A8E0(v129, v15 + 128, v134, v131, &v135);
          if ( (_QWORD *)si128.m128i_i64[0] != v139 )
            j_j___libc_free_0(si128.m128i_u64[0]);
        }
      }
    }
  }
  v25 = *(void (**)())(*(_QWORD *)a1 + 248LL);
  if ( v25 != nullsub_1829 )
    ((void (__fastcall *)(__int64, _QWORD *))v25)(a1, a2);
  v26 = *(_QWORD *)(a1 + 208);
  if ( *(_BYTE *)(v26 + 290) )
  {
    v27 = *(_QWORD *)(a1 + 224);
    if ( *(_BYTE *)(v26 + 21) )
    {
      v28 = a2[25];
      v29 = a2[26];
      strcpy((char *)v139, "ion 20.0.0");
      si128 = _mm_load_si128((const __m128i *)&xmmword_44D4120);
      (*(void (__fastcall **)(__int64, __int64, __int64, __m128i *, __int64, _QWORD, const char *, _QWORD, const char *, _QWORD))(*(_QWORD *)v27 + 640LL))(
        v27,
        v28,
        v29,
        &si128,
        26,
        *(_QWORD *)(*(_QWORD *)v27 + 640LL),
        byte_3F871B3,
        0,
        byte_3F871B3,
        0);
    }
    else
    {
      v100 = *(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v27 + 632LL);
      v101 = sub_C80C60(a2[25], a2[26], 0);
      v100(v27, v101);
    }
  }
  if ( *(_DWORD *)(*(_QWORD *)(a1 + 200) + 564LL) == 8 )
  {
    (*(void (__fastcall **)(__int64, _QWORD *))(*(_QWORD *)a1 + 592LL))(a1, a2);
    (*(void (__fastcall **)(_QWORD, _QWORD, _QWORD))(**(_QWORD **)(a1 + 224) + 176LL))(
      *(_QWORD *)(a1 + 224),
      *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 216) + 168LL) + 24LL),
      0);
    v88 = *(_QWORD **)(a1 + 224);
    v89 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v88[1] + 168LL) + 24LL) + 152LL);
    if ( *(_BYTE *)(v89 + 72) )
      (*(void (__fastcall **)(_QWORD *, __int64, _QWORD, _QWORD))(*v88 + 416LL))(
        v88,
        v89,
        *(_QWORD *)(v89 + 56),
        *(_QWORD *)(v89 + 64));
  }
  v30 = sub_B82360(*(_QWORD *)(a1 + 8), (__int64)&unk_501DA08);
  if ( !v30 )
LABEL_131:
    BUG();
  v132 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v30 + 104LL))(v30, &unk_501DA08);
  v31 = *(__int64 **)(v132 + 176);
  v32 = &v31[*(unsigned int *)(v132 + 184)];
  while ( v32 != v31 )
  {
    while ( 1 )
    {
      v33 = sub_31E4CF0(a1, *v31);
      v34 = v33;
      if ( v33 )
      {
        v35 = *(void (**)())(*(_QWORD *)v33 + 16LL);
        if ( v35 != nullsub_1844 )
          break;
      }
      if ( v32 == ++v31 )
        goto LABEL_31;
    }
    ++v31;
    ((void (__fastcall *)(__int64, _QWORD *, __int64, __int64))v35)(v34, a2, v132, a1);
  }
LABEL_31:
  if ( a2[12] )
  {
    v36 = *(_QWORD *)(a1 + 224);
    v37 = *(void (**)())(*(_QWORD *)v36 + 120LL);
    v141 = 1;
    si128.m128i_i64[0] = (__int64)"Start of file scope inline assembly";
    v140 = 3;
    if ( v37 != nullsub_98 )
    {
      ((void (__fastcall *)(__int64, __m128i *, __int64))v37)(v36, &si128, 1);
      v36 = *(_QWORD *)(a1 + 224);
    }
    v38 = *(void (**)(void))(*(_QWORD *)v36 + 160LL);
    if ( v38 != nullsub_99 )
      v38();
    v39 = *(_QWORD *)(a1 + 200);
    v40 = (_BYTE *)a2[11];
    v41 = *(_QWORD *)(v39 + 656);
    v42 = v39 + 976;
    v43 = *(_QWORD *)(v39 + 680);
    v44 = *(_DWORD *)(v41 + 176);
    v45 = a2[12];
    si128.m128i_i64[0] = (__int64)v139;
    v133 = v43;
    v130 = v44;
    sub_31D5230(si128.m128i_i64, v40, (__int64)&v40[v45]);
    if ( si128.m128i_i64[1] == 0x3FFFFFFFFFFFFFFFLL )
      sub_4262D8((__int64)"basic_string::append");
    sub_2241490((unsigned __int64 *)&si128, "\n", 1u);
    p_si128 = (__m128i *)si128.m128i_i64[0];
    sub_31F20D0(a1, si128.m128i_i32[0], si128.m128i_i32[2], v133, v42, 0, v130);
    v47 = v128;
    if ( (_QWORD *)si128.m128i_i64[0] != v139 )
    {
      p_si128 = (__m128i *)(v139[0] + 1LL);
      j_j___libc_free_0(si128.m128i_u64[0]);
    }
    v48 = *(_QWORD *)(a1 + 224);
    v49 = *(void (**)())(*(_QWORD *)v48 + 120LL);
    v141 = 1;
    si128.m128i_i64[0] = (__int64)"End of file scope inline assembly";
    v140 = 3;
    if ( v49 != nullsub_98 )
    {
      p_si128 = &si128;
      ((void (__fastcall *)(__int64, __m128i *, __int64))v49)(v48, &si128, 1);
      v48 = *(_QWORD *)(a1 + 224);
    }
    v50 = *(void (**)())(*(_QWORD *)v48 + 160LL);
    if ( v50 != nullsub_99 )
      ((void (__fastcall *)(__int64, __m128i *, __int64))v50)(v48, p_si128, v47);
  }
  if ( *(_BYTE *)(*(_QWORD *)(a1 + 208) + 332LL) )
  {
    if ( !(unsigned int)sub_BAA390((__int64)a2) )
      goto LABEL_134;
    if ( *(_DWORD *)(*(_QWORD *)(a1 + 200) + 556LL) == 14 )
    {
      v113 = sub_22077B0(0x5A0u);
      v116 = v113;
      if ( v113 )
        sub_31F7600(v113, a1);
      v117 = *(unsigned int *)(a1 + 584);
      v118 = *(unsigned int *)(a1 + 588);
      si128.m128i_i64[0] = v116;
      v119 = &si128;
      v120 = *(_QWORD *)(a1 + 576);
      v121 = v117 + 1;
      v122 = v117;
      if ( v117 + 1 > v118 )
      {
        v126 = a1 + 576;
        if ( v120 > (unsigned __int64)&si128 || (unsigned __int64)&si128 >= v120 + 8 * v117 )
        {
          sub_31DFD20(v126, v121, v120, v117, v114, v115);
          v117 = *(unsigned int *)(a1 + 584);
          v119 = &si128;
          v120 = *(_QWORD *)(a1 + 576);
          v122 = *(_DWORD *)(a1 + 584);
        }
        else
        {
          v127 = &si128.m128i_i8[-v120];
          sub_31DFD20(v126, v121, v120, v117, v114, v115);
          v120 = *(_QWORD *)(a1 + 576);
          v117 = *(unsigned int *)(a1 + 584);
          v119 = (__m128i *)&v127[v120];
          v122 = *(_DWORD *)(a1 + 584);
        }
      }
      v123 = (_QWORD *)(v120 + 8 * v117);
      if ( v123 )
      {
        *v123 = v119->m128i_i64[0];
        v119->m128i_i64[0] = 0;
        v116 = si128.m128i_i64[0];
        v122 = *(_DWORD *)(a1 + 584);
      }
      *(_DWORD *)(a1 + 584) = v122 + 1;
      if ( v116 )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v116 + 8LL))(v116);
    }
    if ( (unsigned int)sub_BAA300((__int64)a2) )
    {
LABEL_134:
      if ( *(_BYTE *)(a1 + 782) )
      {
        v90 = sub_22077B0(0x18B8u);
        v92 = v90;
        if ( v90 )
          sub_321E420(v90, a1);
        v93 = *(unsigned int *)(a1 + 584);
        v94 = *(unsigned int *)(a1 + 588);
        si128.m128i_i64[0] = v92;
        v95 = &si128;
        *(_QWORD *)(a1 + 760) = v92;
        v96 = *(_QWORD *)(a1 + 576);
        v97 = v93 + 1;
        v98 = v93;
        if ( v93 + 1 > v94 )
        {
          v124 = a1 + 576;
          if ( v96 > (unsigned __int64)&si128 || (unsigned __int64)&si128 >= v96 + 8 * v93 )
          {
            sub_31DFD20(v124, v93 + 1, v93, (__int64)&si128, v97, v91);
            v93 = *(unsigned int *)(a1 + 584);
            v96 = *(_QWORD *)(a1 + 576);
            v95 = &si128;
            v98 = *(_DWORD *)(a1 + 584);
          }
          else
          {
            v125 = &si128.m128i_i8[-v96];
            sub_31DFD20(v124, v93 + 1, v93, (__int64)&si128, v97, v91);
            v96 = *(_QWORD *)(a1 + 576);
            v93 = *(unsigned int *)(a1 + 584);
            v95 = (__m128i *)&v125[v96];
            v98 = *(_DWORD *)(a1 + 584);
          }
        }
        v99 = (_QWORD *)(v96 + 8 * v93);
        if ( v99 )
        {
          *v99 = v95->m128i_i64[0];
          v95->m128i_i64[0] = 0;
          v92 = si128.m128i_i64[0];
          v98 = *(_DWORD *)(a1 + 584);
        }
        *(_DWORD *)(a1 + 584) = v98 + 1;
        if ( v92 )
          (*(void (__fastcall **)(__int64, unsigned __int64, _QWORD *, __m128i *))(*(_QWORD *)v92 + 8LL))(
            v92,
            v96,
            v99,
            v95);
      }
    }
  }
  if ( sub_BA8DC0((__int64)a2, (__int64)"llvm.pseudo_probe_desc", 22) )
  {
    v51 = sub_22077B0(0x28u);
    if ( v51 )
    {
      *(_QWORD *)v51 = a1;
      *(_QWORD *)(v51 + 8) = 0;
      *(_QWORD *)(v51 + 16) = 0;
      *(_QWORD *)(v51 + 24) = 0;
      *(_DWORD *)(v51 + 32) = 0;
    }
    v52 = *(_QWORD *)(a1 + 768);
    *(_QWORD *)(a1 + 768) = v51;
    if ( v52 )
    {
      sub_C7D6A0(*(_QWORD *)(v52 + 16), 24LL * *(unsigned int *)(v52 + 32), 8);
      j_j___libc_free_0(v52);
    }
  }
  v53 = *(_QWORD *)(a1 + 208);
  v54 = *(_DWORD *)(v53 + 336);
  if ( v54 <= 3 )
  {
    v55 = (_QWORD *)a2[4];
    if ( a2 + 3 != v55 )
    {
      while ( 1 )
      {
        v56 = 0;
        if ( v55 )
          v56 = (__int64)(v55 - 7);
        if ( (unsigned int)sub_31DB6B0(a1, v56) )
        {
          v57 = sub_31DB6B0(a1, v56);
          *(_DWORD *)(a1 + 776) = v57;
          if ( v57 == 1 )
          {
LABEL_58:
            v53 = *(_QWORD *)(a1 + 208);
            v54 = *(_DWORD *)(v53 + 336);
            break;
          }
        }
        else if ( *(_DWORD *)(a1 + 776) == 1 )
        {
          goto LABEL_58;
        }
        v55 = (_QWORD *)v55[1];
        if ( a2 + 3 == v55 )
          goto LABEL_58;
      }
    }
  }
  switch ( v54 )
  {
    case 0u:
      if ( sub_31DB810(a1) )
        goto LABEL_61;
      goto LABEL_67;
    case 1u:
    case 2u:
    case 7u:
LABEL_61:
      v58 = sub_22077B0(0x38u);
      v59 = (__int64 *)v58;
      if ( !v58 )
        goto LABEL_67;
      sub_3217550(v58, a1);
      goto LABEL_63;
    case 3u:
      v102 = sub_22077B0(0x20u);
      v59 = (__int64 *)v102;
      if ( !v102 )
        goto LABEL_67;
      sub_3729FE0(v102, a1);
      goto LABEL_63;
    case 4u:
      v104 = *(_DWORD *)(v53 + 344);
      if ( !v104 )
        goto LABEL_67;
      if ( (unsigned int)(v104 - 5) > 1 )
        goto LABEL_131;
      v105 = sub_22077B0(0x48u);
      v59 = (__int64 *)v105;
      if ( v105 )
      {
        sub_3258EA0(v105, a1);
LABEL_63:
        v61 = *(unsigned int *)(a1 + 584);
        v62 = *(unsigned int *)(a1 + 588);
        si128.m128i_i64[0] = (__int64)v59;
        v63 = &si128;
        v64 = *(_QWORD *)(a1 + 576);
        v65 = v61 + 1;
        v66 = v61;
        if ( v61 + 1 > v62 )
        {
          v111 = a1 + 576;
          if ( v64 > (unsigned __int64)&si128 || (unsigned __int64)&si128 >= v64 + 8 * v61 )
          {
            sub_31DFD20(v111, v65, v61, (__int64)&si128, v65, v60);
            v61 = *(unsigned int *)(a1 + 584);
            v64 = *(_QWORD *)(a1 + 576);
            v63 = &si128;
            v66 = *(_DWORD *)(a1 + 584);
          }
          else
          {
            v112 = &si128.m128i_i8[-v64];
            sub_31DFD20(v111, v65, v61, (__int64)&si128, v65, v60);
            v64 = *(_QWORD *)(a1 + 576);
            v61 = *(unsigned int *)(a1 + 584);
            v63 = (__m128i *)&v112[v64];
            v66 = *(_DWORD *)(a1 + 584);
          }
        }
        v67 = (_QWORD *)(v64 + 8 * v61);
        if ( v67 )
        {
          *v67 = v63->m128i_i64[0];
          v63->m128i_i64[0] = 0;
          v68 = si128.m128i_i64[0];
          ++*(_DWORD *)(a1 + 584);
          if ( v68 )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v68 + 8LL))(v68);
        }
        else
        {
          v107 = *v59;
          v108 = (unsigned int)(v66 + 1);
          *(_DWORD *)(a1 + 584) = v108;
          (*(void (__fastcall **)(__int64 *, unsigned __int64, __int64, __m128i *))(v107 + 8))(v59, v64, v108, v63);
        }
      }
LABEL_67:
      v69 = sub_BA91D0((__int64)a2, "cfguard", 7u);
      if ( v69 && *(_QWORD *)(v69 + 136) )
      {
        v70 = sub_22077B0(0x28u);
        v72 = v70;
        if ( v70 )
          sub_3257CC0(v70, a1);
        v73 = *(unsigned int *)(a1 + 560);
        v74 = *(unsigned int *)(a1 + 564);
        si128.m128i_i64[0] = v72;
        v75 = &si128;
        v76 = *(_QWORD *)(a1 + 552);
        v77 = v73 + 1;
        v78 = v73;
        if ( v73 + 1 > v74 )
        {
          v109 = a1 + 552;
          if ( v76 > (unsigned __int64)&si128 || (unsigned __int64)&si128 >= v76 + 8 * v73 )
          {
            sub_31DFD20(v109, v73 + 1, v73, (__int64)&si128, v77, v71);
            v73 = *(unsigned int *)(a1 + 560);
            v76 = *(_QWORD *)(a1 + 552);
            v75 = &si128;
            v78 = *(_DWORD *)(a1 + 560);
          }
          else
          {
            v110 = &si128.m128i_i8[-v76];
            sub_31DFD20(v109, v73 + 1, v73, (__int64)si128.m128i_i64 - v76, v77, v71);
            v76 = *(_QWORD *)(a1 + 552);
            v73 = *(unsigned int *)(a1 + 560);
            v75 = (__m128i *)&v110[v76];
            v78 = *(_DWORD *)(a1 + 560);
          }
        }
        v79 = (_QWORD *)(v76 + 8 * v73);
        if ( v79 )
        {
          *v79 = v75->m128i_i64[0];
          v75->m128i_i64[0] = 0;
          v72 = si128.m128i_i64[0];
          v78 = *(_DWORD *)(a1 + 560);
        }
        *(_DWORD *)(a1 + 560) = v78 + 1;
        if ( v72 )
          (*(void (__fastcall **)(__int64, unsigned __int64, _QWORD *, __m128i *))(*(_QWORD *)v72 + 8LL))(
            v72,
            v76,
            v79,
            v75);
      }
      v80 = *(__int64 **)(a1 + 576);
      v81 = &v80[*(unsigned int *)(a1 + 584)];
      while ( v81 != v80 )
      {
        v82 = *v80++;
        (*(void (__fastcall **)(__int64, _QWORD *))(*(_QWORD *)v82 + 16LL))(v82, a2);
      }
      v83 = *(__int64 **)(a1 + 552);
      v84 = &v83[*(unsigned int *)(a1 + 560)];
      for ( i = v83; v84 != i; ++i )
      {
        v86 = *i;
        (*(void (__fastcall **)(__int64, _QWORD *))(*(_QWORD *)v86 + 16LL))(v86, a2);
      }
      return 0;
    case 5u:
      v106 = sub_22077B0(0x18u);
      v59 = (__int64 *)v106;
      if ( !v106 )
        goto LABEL_67;
      sub_3252AB0(v106, a1);
      *v59 = (__int64)off_4A35F00;
      goto LABEL_63;
    case 6u:
      v103 = sub_22077B0(0x18u);
      v59 = (__int64 *)v103;
      if ( !v103 )
        goto LABEL_67;
      sub_3729450(v103, a1);
      goto LABEL_63;
    default:
      goto LABEL_67;
  }
}
