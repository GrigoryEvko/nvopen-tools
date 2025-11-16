// Function: sub_26CE0F0
// Address: 0x26ce0f0
//
__int64 __fastcall sub_26CE0F0(_QWORD *a1, __int64 *a2, __int64 a3)
{
  _QWORD *v3; // rbx
  __int64 v4; // rcx
  __int64 v5; // rsi
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  __m128i *v10; // rax
  unsigned __int64 v11; // rcx
  __m128i *v12; // rdx
  __int64 v13; // rax
  unsigned __int64 v14; // rdx
  __int64 result; // rax
  __int64 v16; // rax
  __int64 v17; // rdi
  __int64 v18; // r12
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rdi
  __int64 **v22; // r14
  __int64 v23; // r13
  __int64 **v24; // rax
  volatile signed __int32 *v25; // rdi
  unsigned __int64 v26; // r13
  __int64 *v27; // r15
  __int64 *v28; // r14
  __int64 *i; // rdx
  __int64 v30; // rdi
  unsigned int v31; // ecx
  __int64 v32; // rsi
  __int64 *v33; // r15
  unsigned __int64 v34; // r14
  __int64 v35; // rsi
  __int64 v36; // rdi
  unsigned __int64 v37; // rdi
  int v38; // esi
  __int64 v39; // r13
  __int64 v40; // rax
  __int64 v41; // r15
  _QWORD **v42; // r14
  _QWORD *v43; // rdi
  int v44; // edx
  __int64 v45; // r13
  __int64 v46; // rax
  __int64 *v47; // rax
  __int64 v48; // rcx
  __int64 v49; // r14
  __int64 v50; // rbx
  size_t v51; // rsi
  int *v52; // rdi
  _BYTE *v53; // rax
  __int64 v54; // rdx
  __int64 v55; // rdx
  __int64 v56; // rdx
  __int64 v57; // rdx
  __int64 v58; // rdx
  __int64 v59; // rdx
  __int64 *v60; // rsi
  __int64 v61; // rax
  __int64 v62; // r14
  unsigned __int64 v63; // r13
  _QWORD *v64; // r14
  unsigned __int64 v65; // rdi
  unsigned __int64 v66; // rdi
  _QWORD *v67; // r14
  unsigned __int64 v68; // r15
  unsigned __int64 v69; // rdi
  void *v70; // rdi
  char v71; // al
  volatile signed __int32 *v72; // r13
  __int64 v73; // rsi
  __int64 v74; // r8
  __int64 v75; // r9
  int v76; // ecx
  __int64 v77; // r14
  __int64 v78; // r12
  unsigned __int64 v79; // r13
  __int64 v80; // rax
  __int64 v81; // r13
  unsigned __int64 v82; // r14
  unsigned __int64 v83; // rdx
  __int64 v84; // rax
  int *v85; // rdi
  __int64 v86; // rdx
  __int64 v87; // rdx
  __int64 v88; // rdx
  __int64 v89; // rdx
  unsigned int v90; // ecx
  unsigned int v91; // edx
  int v92; // r14d
  unsigned int v93; // eax
  __int64 v94; // rdx
  __int64 v95; // rax
  __int64 v96; // rdi
  __int64 v97; // rdx
  __int64 v98; // r13
  const void *v99; // r15
  size_t v100; // r14
  int v101; // eax
  unsigned int v102; // r12d
  _QWORD *v103; // r8
  __int64 v104; // rax
  _QWORD *v105; // r8
  _QWORD *v106; // rcx
  __int64 v107; // [rsp-10h] [rbp-120h]
  __int64 v108; // [rsp-8h] [rbp-118h]
  __int64 v109; // [rsp-8h] [rbp-118h]
  _QWORD *v110; // [rsp+8h] [rbp-108h]
  int v111; // [rsp+10h] [rbp-100h]
  _QWORD *v112; // [rsp+10h] [rbp-100h]
  __int64 v113; // [rsp+18h] [rbp-F8h]
  _QWORD *v114; // [rsp+20h] [rbp-F0h]
  __int64 v115; // [rsp+20h] [rbp-F0h]
  __int64 *v116; // [rsp+20h] [rbp-F0h]
  __int64 v118; // [rsp+28h] [rbp-E8h]
  __int64 v119; // [rsp+30h] [rbp-E0h]
  unsigned __int8 v121; // [rsp+38h] [rbp-D8h]
  __int64 v122; // [rsp+40h] [rbp-D0h] BYREF
  __int64 v123; // [rsp+48h] [rbp-C8h]
  char v124; // [rsp+50h] [rbp-C0h]
  unsigned __int64 v125[2]; // [rsp+60h] [rbp-B0h] BYREF
  __m128i v126; // [rsp+70h] [rbp-A0h] BYREF
  __int64 v127[4]; // [rsp+80h] [rbp-90h] BYREF
  __int16 v128; // [rsp+A0h] [rbp-70h]
  __int64 **v129; // [rsp+B0h] [rbp-60h] BYREF
  __int64 v130; // [rsp+B8h] [rbp-58h]
  unsigned __int64 v131; // [rsp+C0h] [rbp-50h]
  __int64 v132; // [rsp+C8h] [rbp-48h]
  int v133; // [rsp+D0h] [rbp-40h]
  __int64 *v134; // [rsp+D8h] [rbp-38h]

  v3 = a1;
  v119 = *a2;
  v4 = *a2;
  v5 = a1[151];
  sub_C25870((__int64)&v122, v5, a1[152], v4, a1[159], 0, a1[155], a1[156]);
  v8 = v107;
  v9 = v108;
  if ( (v124 & 1) != 0 )
  {
    v6 = (unsigned int)v122;
    v5 = v123;
    if ( (_DWORD)v122 )
    {
      (*(void (__fastcall **)(__int64 ***, __int64, _QWORD, __int64, __int64, __int64))(*(_QWORD *)v123 + 32LL))(
        &v129,
        v123,
        (unsigned int)v122,
        v7,
        v107,
        v108);
      v10 = (__m128i *)sub_2241130((unsigned __int64 *)&v129, 0, 0, "Could not open profile: ", 0x18u);
      v125[0] = (unsigned __int64)&v126;
      v11 = v10->m128i_i64[0];
      v12 = v10 + 1;
      if ( (__m128i *)v10->m128i_i64[0] != &v10[1] )
      {
LABEL_4:
        v125[0] = v11;
        v126.m128i_i64[0] = v10[1].m128i_i64[0];
LABEL_5:
        v125[1] = v10->m128i_u64[1];
        v10->m128i_i64[0] = (__int64)v12;
        v10->m128i_i64[1] = 0;
        v10[1].m128i_i8[0] = 0;
        sub_2240A30((unsigned __int64 *)&v129);
        v127[0] = (__int64)v125;
        v128 = 260;
        v13 = v3[152];
        v14 = v3[151];
        v130 = 12;
        v133 = 0;
        v132 = v13;
        v134 = v127;
        v129 = (__int64 **)&unk_49D9C78;
        v131 = v14;
        sub_B6EB20(v119, (__int64)&v129);
        sub_2240A30(v125);
        result = 0;
        goto LABEL_6;
      }
LABEL_12:
      v126 = _mm_loadu_si128(v10 + 1);
      goto LABEL_5;
    }
  }
  v16 = v122;
  v17 = a1[142];
  v122 = 0;
  v3[142] = v16;
  if ( v17 )
  {
    (*(void (__fastcall **)(__int64, __int64, __int64, __int64, __int64, __int64))(*(_QWORD *)v17 + 8LL))(
      v17,
      v5,
      v6,
      v7,
      v107,
      v108);
    v16 = v3[142];
  }
  *(_BYTE *)(v16 + 205) = *((_DWORD *)v3 + 380) == 2;
  *(_QWORD *)(v3[142] + 192LL) = a2;
  v18 = v3[142];
  v19 = (*(__int64 (__fastcall **)(__int64, __int64, __int64 *, __int64, __int64, __int64))(*(_QWORD *)v18 + 24LL))(
          v18,
          v5,
          a2,
          v7,
          v8,
          v9);
  if ( (_DWORD)v19 )
  {
    (*(void (__fastcall **)(__int64 ***, __int64, __int64))(*(_QWORD *)v20 + 32LL))(&v129, v20, v19);
    v10 = (__m128i *)sub_2241130((unsigned __int64 *)&v129, 0, 0, "profile reading failed: ", 0x18u);
    v125[0] = (unsigned __int64)&v126;
    v11 = v10->m128i_i64[0];
    v12 = v10 + 1;
    if ( (__m128i *)v10->m128i_i64[0] != &v10[1] )
      goto LABEL_4;
    goto LABEL_12;
  }
  v21 = *(_QWORD *)(v18 + 88);
  if ( v21 )
    sub_C2D850(v21, *(_QWORD *)(v18 + 64));
  unk_4F838D1 = *(_BYTE *)(v18 + 204);
  sub_C1AFD0();
  (*(void (__fastcall **)(__int64 ***))(*(_QWORD *)v3[142] + 40LL))(&v129);
  v22 = v129;
  if ( v129 && (v23 = sub_22077B0(0x18u), v24 = v129, v129 = 0, v23) )
  {
    *(_QWORD *)(v23 + 16) = v24;
    *(_QWORD *)(v23 + 8) = 0x100000001LL;
    *(_QWORD *)v23 = &unk_4A206C8;
    if ( &_pthread_key_create )
      _InterlockedAdd((volatile signed __int32 *)(v23 + 8), 1u);
    else
      ++*(_DWORD *)(v23 + 8);
    sub_A191D0((volatile signed __int32 *)v23);
  }
  else
  {
    v23 = 0;
  }
  v25 = (volatile signed __int32 *)v3[196];
  v3[195] = v22;
  v3[196] = v23;
  if ( v25 )
    sub_A191D0(v25);
  v26 = (unsigned __int64)v129;
  if ( v129 )
  {
    v27 = v129[7];
    v28 = &v27[*((unsigned int *)v129 + 16)];
    if ( v27 != v28 )
    {
      for ( i = v129[7]; ; i = *(__int64 **)(v26 + 56) )
      {
        v30 = *v27;
        v31 = (unsigned int)(v27 - i) >> 7;
        v32 = 4096LL << v31;
        if ( v31 >= 0x1E )
          v32 = 0x40000000000LL;
        ++v27;
        sub_C7D6A0(v30, v32, 16);
        if ( v28 == v27 )
          break;
      }
    }
    v33 = *(__int64 **)(v26 + 104);
    v34 = (unsigned __int64)&v33[2 * *(unsigned int *)(v26 + 112)];
    if ( v33 != (__int64 *)v34 )
    {
      do
      {
        v35 = v33[1];
        v36 = *v33;
        v33 += 2;
        sub_C7D6A0(v36, v35, 16);
      }
      while ( (__int64 *)v34 != v33 );
      v34 = *(_QWORD *)(v26 + 104);
    }
    if ( v34 != v26 + 120 )
      _libc_free(v34);
    v37 = *(_QWORD *)(v26 + 56);
    if ( v37 != v26 + 72 )
      _libc_free(v37);
    sub_C7D6A0(*(_QWORD *)(v26 + 16), 16LL * *(unsigned int *)(v26 + 32), 8);
    j_j___libc_free_0(v26);
  }
  if ( (unsigned int)sub_23DF0D0(&dword_4FF7868) )
    *((_BYTE *)v3 + 1705) = qword_4FF78E8;
  if ( *((_BYTE *)v3 + 1706) )
    sub_26C71D0((__m128i *)(v3[142] + 8LL), *(_BYTE *)(v3[142] + 178LL));
  if ( !(_BYTE)qword_4FF7D48 || !v3[195] || (_BYTE)qword_4FF7F08 )
  {
    *((_BYTE *)v3 + 1704) = 0;
    goto LABEL_66;
  }
  v38 = *((_DWORD *)v3 + 415);
  *((_BYTE *)v3 + 1704) = 1;
  if ( v38 )
  {
    v39 = 0;
    v40 = *((unsigned int *)v3 + 414);
    v41 = 8 * v40;
    if ( (_DWORD)v40 )
    {
      do
      {
        v42 = (_QWORD **)(v39 + v3[206]);
        v43 = *v42;
        if ( *v42 != (_QWORD *)-8LL && v43 )
          sub_C7D6A0((__int64)v43, *v43 + 9LL, 8);
        v39 += 8;
        *v42 = 0;
      }
      while ( v39 != v41 );
    }
    *(_QWORD *)((char *)v3 + 1660) = 0;
  }
  v44 = *((_DWORD *)v3 + 422);
  ++v3[209];
  v45 = (__int64)(v3 + 209);
  if ( v44 )
  {
    v90 = 4 * v44;
    v46 = *((unsigned int *)v3 + 424);
    if ( (unsigned int)(4 * v44) < 0x40 )
      v90 = 64;
    if ( v90 >= (unsigned int)v46 )
      goto LABEL_56;
    v91 = v44 - 1;
    if ( v91 )
    {
      _BitScanReverse(&v91, v91);
      v92 = 1 << (33 - (v91 ^ 0x1F));
      if ( v92 < 64 )
        v92 = 64;
      if ( v92 == (_DWORD)v46 )
        goto LABEL_148;
    }
    else
    {
      v92 = 64;
    }
    sub_C7D6A0(v3[210], 8 * v46, 8);
    v93 = sub_26BC060(v92);
    *((_DWORD *)v3 + 424) = v93;
    if ( !v93 )
      goto LABEL_167;
    v3[210] = sub_C7D670(8LL * v93, 8);
LABEL_148:
    sub_D7A370((__int64)(v3 + 209));
    goto LABEL_59;
  }
  if ( *((_DWORD *)v3 + 423) )
  {
    v46 = *((unsigned int *)v3 + 424);
    if ( (unsigned int)v46 <= 0x40 )
    {
LABEL_56:
      if ( 8LL * (unsigned int)v46 )
        memset((void *)v3[210], 255, 8LL * (unsigned int)v46);
      goto LABEL_58;
    }
    sub_C7D6A0(v3[210], 8 * v46, 8);
    *((_DWORD *)v3 + 424) = 0;
LABEL_167:
    v3[210] = 0;
LABEL_58:
    v3[211] = 0;
  }
LABEL_59:
  v47 = (__int64 *)(*(__int64 (__fastcall **)(_QWORD))(*(_QWORD *)v3[142] + 48LL))(v3[142]);
  if ( v47 )
  {
    v48 = *v47;
    v113 = v47[1];
    if ( unk_4F838D1 )
    {
      if ( v48 != v47[1] )
      {
        v114 = v3;
        v49 = v47[1];
        v50 = *v47;
        do
        {
          v51 = *(_QWORD *)(v50 + 8);
          v52 = *(int **)v50;
          v50 += 16;
          v127[0] = sub_26BA4C0(v52, v51);
          sub_D7AC80((__int64)&v129, v45, v127);
        }
        while ( v49 != v50 );
        v3 = v114;
      }
      goto LABEL_65;
    }
    if ( v48 != v113 )
    {
      v98 = *v47;
      v116 = v3 + 206;
      while ( 1 )
      {
        v99 = *(const void **)v98;
        v100 = *(_QWORD *)(v98 + 8);
        if ( !*(_QWORD *)v98 )
          v100 = 0;
        v101 = sub_C92610();
        v102 = sub_C92740((__int64)v116, v99, v100, v101);
        v103 = (_QWORD *)(v3[206] + 8LL * v102);
        if ( !*v103 )
          goto LABEL_162;
        if ( *v103 == -8 )
          break;
LABEL_156:
        v98 += 16;
        if ( v113 == v98 )
          goto LABEL_65;
      }
      --*((_DWORD *)v3 + 416);
LABEL_162:
      v112 = v103;
      v104 = sub_C7D670(v100 + 9, 8);
      v105 = v112;
      v106 = (_QWORD *)v104;
      if ( v100 )
      {
        v110 = (_QWORD *)v104;
        memcpy((void *)(v104 + 8), v99, v100);
        v105 = v112;
        v106 = v110;
      }
      *((_BYTE *)v106 + v100 + 8) = 0;
      *v106 = v100;
      *v105 = v106;
      ++*((_DWORD *)v3 + 415);
      sub_C929D0(v116, v102);
      goto LABEL_156;
    }
  }
LABEL_65:
  *((_BYTE *)v3 + 1128) = 1;
LABEL_66:
  if ( a3 && qword_4FF6AD0 )
  {
    v94 = *((unsigned int *)v3 + 380);
    v130 = qword_4FF6AD0;
    v125[0] = 0;
    v131 = __PAIR64__(dword_4FF6608, dword_4FF6868);
    v129 = (__int64 **)qword_4FF6AC8;
    LODWORD(v132) = dword_4FF63A8;
    sub_310A360(
      (unsigned int)v127,
      (_DWORD)a2,
      a3,
      v119,
      (unsigned int)v125,
      (unsigned int)&v129,
      0,
      v94 | 0x600000000LL);
    v95 = v127[0];
    v96 = v3[214];
    v127[0] = 0;
    v3[214] = v95;
    v97 = v109;
    if ( v96 )
    {
      (*(void (__fastcall **)(__int64, __int64 *, __int64))(*(_QWORD *)v96 + 8LL))(v96, a2, v109);
      if ( v127[0] )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v127[0] + 8LL))(v127[0]);
    }
    if ( v125[0] )
      (*(void (__fastcall **)(unsigned __int64, __int64 *, __int64))(*(_QWORD *)v125[0] + 8LL))(v125[0], a2, v97);
  }
  v53 = (_BYTE *)v3[142];
  if ( !v53[178] && !v53[179] && !v53[177] )
    goto LABEL_102;
  if ( !(unsigned int)sub_23DF0D0(&dword_4F8E408) )
  {
    LOBYTE(qword_4F8E448[8]) = 1;
    LOBYTE(v129) = 1;
    sub_26C3780((__int64)&qword_4F8E448[12], (__int64)&v129, v54);
  }
  if ( !(unsigned int)sub_23DF0D0(&dword_500B9E8) )
  {
    LOBYTE(qword_500BA28[8]) = 1;
    LOBYTE(v129) = 1;
    sub_26C3780((__int64)&qword_500BA28[12], (__int64)&v129, v55);
  }
  if ( !(unsigned int)sub_23DF0D0(&dword_5008A28) )
  {
    LOBYTE(qword_5008A68[8]) = 1;
    LOBYTE(v129) = 1;
    sub_26C3780((__int64)&qword_5008A68[12], (__int64)&v129, v56);
  }
  if ( !(unsigned int)sub_23DF0D0(dword_4FF7948) )
  {
    LOBYTE(v129) = 1;
    byte_4FF79C8 = 1;
    sub_26C3780((__int64)&unk_4FF79E8, (__int64)&v129, v57);
  }
  if ( !(unsigned int)sub_23DF0D0(&dword_4FF6DE8) )
  {
    LOBYTE(v129) = 1;
    LOBYTE(qword_4FF6E68) = 1;
    sub_26C3780((__int64)&unk_4FF6E88, (__int64)&v129, v58);
  }
  if ( !(unsigned int)sub_23DF0D0(&dword_4FF6C28) )
  {
    LOBYTE(v129) = 1;
    LOBYTE(qword_4FF6CA8) = 1;
    sub_26C3780((__int64)&unk_4FF6CC8, (__int64)&v129, v59);
  }
  v60 = (__int64 *)v3[142];
  if ( *((_BYTE *)v60 + 179) )
  {
    if ( !(unsigned int)sub_23DF0D0(&dword_4FF6D08) )
    {
      LOBYTE(v129) = 1;
      LOBYTE(qword_4FF6D88) = 1;
      sub_26C3780((__int64)&unk_4FF6DA8, (__int64)&v129, v89);
    }
    v60 = (__int64 *)v3[142];
  }
  if ( *((_BYTE *)v60 + 177) )
  {
    if ( !(unsigned int)sub_23DF0D0((int *)&qword_4FF8200[1]) )
    {
      LOBYTE(qword_4FF8200[17]) = 1;
      LOBYTE(v129) = 1;
      sub_26C3780((__int64)&qword_4FF8200[21], (__int64)&v129, v87);
    }
    if ( !(unsigned int)sub_23DF0D0((int *)&qword_4FF8120[1]) )
    {
      LOBYTE(qword_4FF8120[17]) = 1;
      LOBYTE(v129) = 1;
      sub_26C3780((__int64)&qword_4FF8120[21], (__int64)&v129, v88);
    }
    v60 = (__int64 *)v3[142];
  }
  if ( *((_BYTE *)v60 + 178) )
    goto LABEL_86;
  v85 = dword_4FF75C8;
  if ( !(unsigned int)sub_23DF0D0(dword_4FF75C8) )
  {
    dword_4FF75C8[32] = -1;
    LODWORD(v127[0]) = -1;
    if ( !*(_QWORD *)&dword_4FF75C8[44] )
      goto LABEL_176;
    v60 = v127;
    (*(void (__fastcall **)(int *, __int64 *))&dword_4FF75C8[46])(&dword_4FF75C8[40], v127);
  }
  v85 = dword_4FF74E8;
  if ( (unsigned int)sub_23DF0D0(dword_4FF74E8) )
    goto LABEL_129;
  LODWORD(v129) = -1;
  dword_4FF74E8[32] = -1;
  if ( !*(_QWORD *)&dword_4FF74E8[44] )
LABEL_176:
    sub_4263D6(v85, v60, v86);
  (*(void (__fastcall **)(int *, __int64 ***))&dword_4FF74E8[46])(&dword_4FF74E8[40], &v129);
LABEL_129:
  v60 = (__int64 *)v3[142];
  if ( !*((_BYTE *)v60 + 178) )
  {
    v71 = *((_BYTE *)v60 + 177);
    goto LABEL_101;
  }
LABEL_86:
  v61 = sub_22077B0(0xD8u);
  v62 = v61;
  if ( v61 )
    sub_3180150(v61, v60 + 1, v3 + 202);
  v63 = v3[189];
  v3[189] = v62;
  if ( v63 )
  {
    sub_26BC0B0(*(_QWORD **)(v63 + 136));
    v64 = *(_QWORD **)(v63 + 72);
    while ( v64 )
    {
      v65 = (unsigned __int64)v64;
      v64 = (_QWORD *)*v64;
      j_j___libc_free_0(v65);
    }
    memset(*(void **)(v63 + 56), 0, 8LL * *(_QWORD *)(v63 + 64));
    v66 = *(_QWORD *)(v63 + 56);
    *(_QWORD *)(v63 + 80) = 0;
    *(_QWORD *)(v63 + 72) = 0;
    if ( v66 != v63 + 104 )
      j_j___libc_free_0(v66);
    v67 = *(_QWORD **)(v63 + 16);
    while ( v67 )
    {
      v68 = (unsigned __int64)v67;
      v67 = (_QWORD *)*v67;
      v69 = *(_QWORD *)(v68 + 16);
      if ( v69 )
        j_j___libc_free_0(v69);
      j_j___libc_free_0(v68);
    }
    memset(*(void **)v63, 0, 8LL * *(_QWORD *)(v63 + 8));
    v70 = *(void **)v63;
    *(_QWORD *)(v63 + 24) = 0;
    *(_QWORD *)(v63 + 16) = 0;
    if ( v70 != (void *)(v63 + 48) )
      j_j___libc_free_0((unsigned __int64)v70);
    j_j___libc_free_0(v63);
  }
  v71 = *(_BYTE *)(v3[142] + 177LL);
LABEL_101:
  if ( v71 )
  {
    v80 = sub_22077B0(0x20u);
    v81 = v80;
    if ( v80 )
      sub_26C9BA0(v80, (__int64)a2);
    v82 = v3[149];
    v3[149] = v81;
    if ( v82 )
    {
      sub_C7D6A0(*(_QWORD *)(v82 + 8), 24LL * *(unsigned int *)(v82 + 24), 8);
      j_j___libc_free_0(v82);
    }
    if ( !sub_BA8DC0((__int64)a2, (__int64)"llvm.pseudo_probe_desc", 22) )
    {
      v127[0] = (__int64)"Pseudo-probe-based profile requires SampleProfileProbePass";
      v83 = a2[21];
      v84 = a2[22];
      v128 = 259;
      v130 = 0x10000000CLL;
      v132 = v84;
      v134 = v127;
      v129 = (__int64 **)&unk_49D9C78;
      v131 = v83;
      v133 = 0;
      sub_B6EB20(v119, (__int64)&v129);
      result = 0;
      goto LABEL_6;
    }
  }
LABEL_102:
  if ( LOBYTE(qword_4FF8040[17]) || LOBYTE(qword_4FF7F60[17]) || (result = 1, LOBYTE(qword_4FF8200[17])) )
  {
    v72 = (volatile signed __int32 *)v3[196];
    v73 = v3[149];
    v74 = v3[188];
    v75 = v3[142];
    v76 = *((_DWORD *)v3 + 380);
    v77 = v3[195];
    if ( v72 )
    {
      if ( &_pthread_key_create )
        _InterlockedAdd(v72 + 2, 1u);
      else
        ++*((_DWORD *)v72 + 2);
    }
    v111 = v76;
    v115 = v75;
    v118 = v74;
    v78 = sub_22077B0(0x1B0u);
    if ( v78 )
    {
      *(_QWORD *)(v78 + 48) = 1;
      *(_QWORD *)(v78 + 56) = 0;
      *(_QWORD *)v78 = a2;
      *(_QWORD *)(v78 + 40) = v78 + 88;
      *(_QWORD *)(v78 + 112) = 0x4000000000LL;
      *(_QWORD *)(v78 + 136) = 0x4000000000LL;
      *(_QWORD *)(v78 + 144) = v78 + 192;
      *(_QWORD *)(v78 + 200) = v78 + 248;
      *(_QWORD *)(v78 + 8) = v115;
      *(_QWORD *)(v78 + 16) = v118;
      *(_QWORD *)(v78 + 24) = v73;
      *(_DWORD *)(v78 + 32) = v111;
      *(_QWORD *)(v78 + 64) = 0;
      *(_QWORD *)(v78 + 80) = 0;
      *(_QWORD *)(v78 + 88) = 0;
      *(_QWORD *)(v78 + 96) = 0;
      *(_QWORD *)(v78 + 104) = 0;
      *(_QWORD *)(v78 + 120) = 0;
      *(_QWORD *)(v78 + 128) = 0;
      *(_QWORD *)(v78 + 152) = 1;
      *(_QWORD *)(v78 + 160) = 0;
      *(_QWORD *)(v78 + 168) = 0;
      *(_QWORD *)(v78 + 184) = 0;
      *(_QWORD *)(v78 + 192) = 0;
      *(_DWORD *)(v78 + 72) = 1065353216;
      *(_DWORD *)(v78 + 176) = 1065353216;
      *(_QWORD *)(v78 + 208) = 1;
      *(_QWORD *)(v78 + 216) = 0;
      *(_QWORD *)(v78 + 224) = 0;
      *(_QWORD *)(v78 + 240) = 0;
      *(_QWORD *)(v78 + 248) = 0;
      *(_QWORD *)(v78 + 256) = v3 + 169;
      *(_QWORD *)(v78 + 264) = v3 + 162;
      *(_QWORD *)(v78 + 272) = v78 + 320;
      *(_QWORD *)(v78 + 280) = 1;
      *(_QWORD *)(v78 + 288) = 0;
      *(_QWORD *)(v78 + 296) = 0;
      *(_QWORD *)(v78 + 312) = 0;
      *(_QWORD *)(v78 + 320) = 0;
      *(_QWORD *)(v78 + 328) = v77;
      *(_QWORD *)(v78 + 336) = v72;
      *(_DWORD *)(v78 + 232) = 1065353216;
      *(_DWORD *)(v78 + 304) = 1065353216;
      if ( v72 )
      {
        if ( &_pthread_key_create )
          _InterlockedAdd(v72 + 2, 1u);
        else
          ++*((_DWORD *)v72 + 2);
      }
      *(_QWORD *)(v78 + 344) = 0;
      *(_QWORD *)(v78 + 352) = 0;
      *(_QWORD *)(v78 + 360) = 0;
      *(_QWORD *)(v78 + 368) = 0;
      *(_QWORD *)(v78 + 376) = 0;
      *(_QWORD *)(v78 + 384) = 0;
      *(_QWORD *)(v78 + 392) = 0;
      *(_QWORD *)(v78 + 400) = 0;
      *(_QWORD *)(v78 + 408) = 0;
      *(_QWORD *)(v78 + 416) = 0;
      *(_QWORD *)(v78 + 424) = 0;
    }
    if ( v72 )
      sub_A191D0(v72);
    v79 = v3[215];
    v3[215] = v78;
    if ( v79 )
    {
      sub_26C2E30(v79);
      j_j___libc_free_0(v79);
    }
    result = 1;
  }
LABEL_6:
  if ( (v124 & 1) == 0 )
  {
    if ( v122 )
    {
      v121 = result;
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v122 + 8LL))(v122);
      return v121;
    }
  }
  return result;
}
