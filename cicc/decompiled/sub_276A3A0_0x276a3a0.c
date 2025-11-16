// Function: sub_276A3A0
// Address: 0x276a3a0
//
__int64 __fastcall sub_276A3A0(_QWORD *a1, __int64 a2, _QWORD *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r12
  __int64 v8; // r14
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __m128i v14; // xmm1
  __m128i v15; // xmm2
  __m128i v16; // xmm3
  _BYTE *v17; // rsi
  __int64 v18; // rbx
  _QWORD *v19; // rax
  unsigned int v20; // eax
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rdx
  _BYTE **v25; // rax
  _BYTE **v26; // rdx
  _BYTE *v27; // r14
  unsigned __int64 v28; // rax
  __int64 v29; // r15
  char v30; // r14
  unsigned int v31; // r13d
  __int64 *v32; // rdx
  __int64 v33; // rcx
  __int64 v34; // r8
  __int64 v35; // r9
  __int64 v36; // r10
  __int64 *v37; // rax
  _QWORD *v38; // rcx
  _QWORD *v39; // rdx
  _QWORD *v40; // rax
  __int64 v41; // rsi
  char v42; // dl
  _QWORD *v43; // rax
  _QWORD *v44; // rdx
  __int64 *v45; // rax
  __int64 v46; // rax
  int v47; // edx
  __int64 v48; // rdi
  int v49; // edx
  unsigned int v50; // ecx
  _QWORD *v51; // rax
  _BYTE *v52; // r9
  __int64 v53; // rsi
  unsigned int v54; // ecx
  __int64 *v55; // rax
  __int64 v56; // r9
  __int64 v57; // r13
  unsigned __int64 *v58; // r12
  __int64 v59; // r14
  __int64 v60; // rbx
  __int64 *v61; // rdi
  __int64 *v62; // rdi
  unsigned __int64 v63; // rdx
  _QWORD **v64; // rax
  char *v65; // r8
  char *v66; // rsi
  _QWORD *v67; // rcx
  void *v68; // rdi
  size_t v69; // rdx
  __int64 *v70; // r14
  void *v71; // rax
  __int64 v72; // rcx
  __int64 v73; // r8
  __int64 v74; // rdi
  __int64 v75; // rsi
  void *v76; // rdx
  __int64 v77; // r11
  __int64 v78; // r10
  __int64 v79; // r9
  __int64 v80; // rax
  __int64 *v81; // rax
  int i; // eax
  int v83; // r8d
  __int64 v84; // rax
  int v85; // eax
  int v86; // r8d
  __int64 v87; // rax
  unsigned int v88; // [rsp+0h] [rbp-410h]
  void *v89; // [rsp+0h] [rbp-410h]
  __int64 v90; // [rsp+8h] [rbp-408h]
  __int64 v91; // [rsp+10h] [rbp-400h]
  __int64 v92; // [rsp+10h] [rbp-400h]
  int v93; // [rsp+18h] [rbp-3F8h]
  int v94; // [rsp+1Ch] [rbp-3F4h]
  __int64 v95; // [rsp+20h] [rbp-3F0h]
  __int64 *v96; // [rsp+20h] [rbp-3F0h]
  _BYTE *v97; // [rsp+28h] [rbp-3E8h] BYREF
  _BYTE *v98; // [rsp+30h] [rbp-3E0h] BYREF
  __int64 v99; // [rsp+38h] [rbp-3D8h]
  _QWORD v100[2]; // [rsp+40h] [rbp-3D0h] BYREF
  __int64 *v101; // [rsp+50h] [rbp-3C0h]
  __int64 v102; // [rsp+58h] [rbp-3B8h]
  __int64 v103; // [rsp+60h] [rbp-3B0h] BYREF
  __m128i v104; // [rsp+70h] [rbp-3A0h] BYREF
  __int64 *v105; // [rsp+80h] [rbp-390h] BYREF
  __int64 *v106; // [rsp+88h] [rbp-388h]
  __int64 v107; // [rsp+90h] [rbp-380h] BYREF
  __m128i v108; // [rsp+98h] [rbp-378h] BYREF
  __int64 v109; // [rsp+A8h] [rbp-368h]
  __m128i v110; // [rsp+B0h] [rbp-360h] BYREF
  __m128i v111; // [rsp+C0h] [rbp-350h]
  _QWORD v112[2]; // [rsp+D0h] [rbp-340h] BYREF
  _BYTE v113[324]; // [rsp+E0h] [rbp-330h] BYREF
  int v114; // [rsp+224h] [rbp-1ECh]
  __int64 v115; // [rsp+228h] [rbp-1E8h]
  __m128i v116; // [rsp+230h] [rbp-1E0h] BYREF
  __m128i v117; // [rsp+240h] [rbp-1D0h] BYREF
  __m128i v118; // [rsp+250h] [rbp-1C0h] BYREF
  void *dest[2]; // [rsp+260h] [rbp-1B0h] BYREF
  __m128i v120; // [rsp+270h] [rbp-1A0h] BYREF
  char v121[8]; // [rsp+280h] [rbp-190h] BYREF
  int v122; // [rsp+288h] [rbp-188h]
  char v123; // [rsp+3D0h] [rbp-40h]
  int v124; // [rsp+3D4h] [rbp-3Ch]
  __int64 v125; // [rsp+3D8h] [rbp-38h]

  v7 = (__int64)a1;
  *a1 = 0;
  a1[1] = 0;
  a1[2] = 0;
  v97 = a3;
  v95 = a4;
  v94 = a6;
  if ( (unsigned int)a6 > (unsigned int)qword_4FFAEE8 )
  {
    v8 = **(_QWORD **)(a2 + 24);
    v96 = *(__int64 **)(a2 + 24);
    v9 = sub_B2BE50(v8);
    if ( sub_B6EA50(v9)
      || (v22 = sub_B2BE50(v8),
          v23 = sub_B6F970(v22),
          (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v23 + 48LL))(v23)) )
    {
      sub_B178C0(
        (__int64)&v116,
        (__int64)"dfa-jump-threading",
        (__int64)"MaxPathLengthReached",
        20,
        *(_QWORD *)(a2 + 8));
      sub_B18290((__int64)&v116, "Exploration stopped after visiting MaxPathLength=", 0x31u);
      sub_B169E0((__int64 *)&v98, "MaxPathLength", 13, qword_4FFAEE8);
      v105 = &v107;
      sub_2765450((__int64 *)&v105, v98, (__int64)&v98[v99]);
      v108.m128i_i64[1] = (__int64)&v110;
      sub_2765450(&v108.m128i_i64[1], v101, (__int64)v101 + v102);
      v111 = _mm_loadu_si128(&v104);
      sub_B180C0((__int64)&v116, (unsigned __int64)&v105);
      if ( (__m128i *)v108.m128i_i64[1] != &v110 )
        j_j___libc_free_0(v108.m128i_u64[1]);
      if ( v105 != &v107 )
        j_j___libc_free_0((unsigned __int64)v105);
      sub_B18290((__int64)&v116, " blocks.", 8u);
      v14 = _mm_loadu_si128((const __m128i *)&v117.m128i_u64[1]);
      v15 = _mm_loadu_si128((const __m128i *)dest);
      LODWORD(v106) = v116.m128i_i32[2];
      v16 = _mm_loadu_si128(&v120);
      v108 = v14;
      BYTE4(v106) = v116.m128i_i8[12];
      v110 = v15;
      v107 = v117.m128i_i64[0];
      v105 = (__int64 *)&unk_49D9D40;
      v111 = v16;
      v109 = v118.m128i_i64[1];
      v112[0] = v113;
      v112[1] = 0x400000000LL;
      if ( v122 )
        sub_2768F90((__int64)v112, (__int64)v121, v10, v11, v12, v13);
      v113[320] = v123;
      v114 = v124;
      v115 = v125;
      v105 = (__int64 *)&unk_49D9DE8;
      if ( v101 != &v103 )
        j_j___libc_free_0((unsigned __int64)v101);
      if ( v98 != (_BYTE *)v100 )
        j_j___libc_free_0((unsigned __int64)v98);
      v116.m128i_i64[0] = (__int64)&unk_49D9D40;
      sub_23FD590((__int64)v121);
      sub_1049740(v96, (__int64)&v105);
      v105 = (__int64 *)&unk_49D9D40;
      sub_23FD590((__int64)v112);
    }
    return v7;
  }
  v17 = v97;
  v18 = a5;
  if ( !*(_BYTE *)(a5 + 28) )
    goto LABEL_21;
  v19 = *(_QWORD **)(a5 + 8);
  a4 = *(unsigned int *)(a5 + 20);
  a3 = &v19[a4];
  if ( v19 != a3 )
  {
    while ( v97 != (_BYTE *)*v19 )
    {
      if ( a3 == ++v19 )
        goto LABEL_24;
    }
    goto LABEL_19;
  }
LABEL_24:
  if ( (unsigned int)a4 < *(_DWORD *)(a5 + 16) )
  {
    *(_DWORD *)(a5 + 20) = a4 + 1;
    *a3 = v17;
    ++*(_QWORD *)a5;
  }
  else
  {
LABEL_21:
    sub_C8CC70(a5, (__int64)v97, (__int64)a3, a4, a5, a6);
  }
LABEL_19:
  v20 = *(_DWORD *)a2 + 1;
  *(_DWORD *)a2 = v20;
  if ( v20 > (unsigned int)qword_4FFAE08 )
    return v7;
  v24 = *(_QWORD *)(a2 + 64);
  if ( *(_BYTE *)(v24 + 84) )
  {
    v25 = *(_BYTE ***)(v24 + 64);
    v26 = &v25[*(unsigned int *)(v24 + 76)];
    if ( v25 == v26 )
      return v7;
    while ( 1 )
    {
      v27 = *v25;
      if ( v97 == *v25 )
        break;
      if ( v26 == ++v25 )
        return v7;
    }
  }
  else
  {
    if ( !sub_C8CA60(v24 + 56, (__int64)v97) )
      return v7;
    v27 = v97;
  }
  v105 = 0;
  v106 = &v108.m128i_i64[1];
  v107 = 4;
  v108.m128i_i32[0] = 0;
  v108.m128i_i8[4] = 1;
  v28 = *((_QWORD *)v27 + 6) & 0xFFFFFFFFFFFFFFF8LL;
  if ( (_BYTE *)v28 == v27 + 48 )
    goto LABEL_46;
  if ( !v28 )
    BUG();
  v29 = v28 - 24;
  if ( (unsigned int)*(unsigned __int8 *)(v28 - 24) - 30 > 0xA || (v93 = sub_B46E30(v29)) == 0 )
  {
LABEL_46:
    if ( *(_BYTE *)(v18 + 28) )
    {
      v38 = *(_QWORD **)(v18 + 8);
      v39 = &v38[*(unsigned int *)(v18 + 20)];
      v40 = v38;
      if ( v38 != v39 )
      {
        while ( (_BYTE *)*v40 != v27 )
        {
          if ( v39 == ++v40 )
            goto LABEL_52;
        }
        v41 = (unsigned int)(*(_DWORD *)(v18 + 20) - 1);
        *(_DWORD *)(v18 + 20) = v41;
        *v40 = v38[v41];
        ++*(_QWORD *)v18;
      }
    }
    else
    {
      v81 = sub_C8CA60(v18, (__int64)v27);
      if ( v81 )
      {
        *v81 = -2;
        ++*(_DWORD *)(v18 + 24);
        ++*(_QWORD *)v18;
      }
    }
    goto LABEL_52;
  }
  v90 = a2;
  v30 = 1;
  v31 = 0;
  while ( 1 )
  {
    v36 = sub_B46EC0(v29, v31);
    if ( !v30 )
      goto LABEL_54;
    v37 = v106;
    v33 = HIDWORD(v107);
    v32 = &v106[HIDWORD(v107)];
    if ( v106 != v32 )
    {
      while ( v36 != *v37 )
      {
        if ( v32 == ++v37 )
          goto LABEL_79;
      }
      goto LABEL_41;
    }
LABEL_79:
    if ( HIDWORD(v107) < (unsigned int)v107 )
    {
      ++HIDWORD(v107);
      *v32 = v36;
      v105 = (__int64 *)((char *)v105 + 1);
    }
    else
    {
LABEL_54:
      v91 = v36;
      sub_C8CC70((__int64)&v105, v36, (__int64)v32, v33, v34, v35);
      v36 = v91;
      if ( !v42 )
        goto LABEL_41;
    }
    if ( v36 == v95 )
    {
      v116 = 0u;
      v98 = v97;
      v117 = 0u;
      v99 = v95;
      v118 = 0u;
      dest[0] = 0;
      dest[1] = 0;
      v120 = 0u;
      sub_2768EA0(v116.m128i_i64, 2u);
      v64 = (_QWORD **)v118.m128i_i64[1];
      if ( v118.m128i_i64[1] >= (unsigned __int64)v120.m128i_i64[1] )
      {
        v68 = dest[1];
        v69 = 16;
        v65 = (char *)&v98;
      }
      else
      {
        v65 = (char *)&v98;
        do
        {
          v66 = v65;
          v67 = *v64++;
          v65 += 512;
          *v67 = *(_QWORD *)v66;
          v67[63] = *((_QWORD *)v66 + 63);
          qmemcpy(
            (void *)((unsigned __int64)(v67 + 1) & 0xFFFFFFFFFFFFFFF8LL),
            (const void *)(v66 - ((char *)v67 - ((unsigned __int64)(v67 + 1) & 0xFFFFFFFFFFFFFFF8LL))),
            8LL * (((unsigned int)v67 - (((_DWORD)v67 + 8) & 0xFFFFFFF8) + 512) >> 3));
        }
        while ( v120.m128i_i64[1] > (unsigned __int64)v64 );
        if ( v65 == (char *)v100 )
          goto LABEL_87;
        v68 = dest[1];
        v69 = (char *)v100 - v65;
      }
      memcpy(v68, v65, v69);
LABEL_87:
      v70 = *(__int64 **)(v7 + 8);
      if ( v70 == *(__int64 **)(v7 + 16) )
      {
        sub_2769210(v7, *(_BYTE **)(v7 + 8), &v116);
      }
      else
      {
        if ( v70 )
        {
          *v70 = 0;
          v70[1] = 0;
          v70[2] = 0;
          v70[3] = 0;
          v70[4] = 0;
          v70[5] = 0;
          v70[6] = 0;
          v70[7] = 0;
          v70[8] = 0;
          v70[9] = 0;
          sub_2768EA0(v70, 0);
          if ( v116.m128i_i64[0] )
          {
            v71 = (void *)v70[7];
            v72 = v70[5];
            v70[7] = 0;
            v73 = v70[2];
            v74 = v70[3];
            v70[2] = 0;
            v75 = v70[4];
            v76 = (void *)v70[6];
            v70[3] = 0;
            v77 = v70[8];
            v78 = *v70;
            v70[4] = 0;
            *v70 = 0;
            v79 = v70[1];
            v70[5] = 0;
            v70[1] = 0;
            v70[6] = 0;
            v70[8] = 0;
            v89 = v71;
            v80 = v70[9];
            v70[9] = 0;
            *(__m128i *)v70 = _mm_loadu_si128(&v116);
            *((__m128i *)v70 + 1) = _mm_loadu_si128(&v117);
            *((__m128i *)v70 + 2) = _mm_loadu_si128(&v118);
            *((__m128i *)v70 + 3) = _mm_loadu_si128((const __m128i *)dest);
            *((__m128i *)v70 + 4) = _mm_loadu_si128(&v120);
            v116.m128i_i64[0] = v78;
            v116.m128i_i64[1] = v79;
            v118.m128i_i64[1] = v72;
            v117.m128i_i64[0] = v73;
            v117.m128i_i64[1] = v74;
            v118.m128i_i64[0] = v75;
            dest[0] = v76;
            dest[1] = v89;
            v120.m128i_i64[0] = v77;
            v120.m128i_i64[1] = v80;
          }
          v70 = *(__int64 **)(v7 + 8);
        }
        *(_QWORD *)(v7 + 8) = v70 + 10;
      }
      sub_2767770((unsigned __int64 *)&v116);
      goto LABEL_41;
    }
    if ( !*(_BYTE *)(v18 + 28) )
      break;
    v43 = *(_QWORD **)(v18 + 8);
    v44 = &v43[*(unsigned int *)(v18 + 20)];
    if ( v43 == v44 )
      goto LABEL_63;
    while ( v36 != *v43 )
    {
      if ( v44 == ++v43 )
        goto LABEL_63;
    }
LABEL_41:
    if ( v93 == ++v31 )
    {
      v27 = v97;
      goto LABEL_46;
    }
    v30 = v108.m128i_i8[4];
  }
  v92 = v36;
  v45 = sub_C8CA60(v18, v36);
  v36 = v92;
  if ( v45 )
    goto LABEL_41;
LABEL_63:
  v46 = *(_QWORD *)(v90 + 56);
  v47 = *(_DWORD *)(v46 + 24);
  v48 = *(_QWORD *)(v46 + 8);
  if ( !v47 )
    BUG();
  v49 = v47 - 1;
  v50 = v49 & (((unsigned int)v97 >> 9) ^ ((unsigned int)v97 >> 4));
  v51 = (_QWORD *)(v48 + 16LL * v50);
  v52 = (_BYTE *)*v51;
  if ( v97 != (_BYTE *)*v51 )
  {
    for ( i = 1; ; i = v83 )
    {
      if ( v52 == (_BYTE *)-4096LL )
        BUG();
      v83 = i + 1;
      v84 = v49 & (v50 + i);
      v50 = v84;
      v51 = (_QWORD *)(v48 + 16 * v84);
      v52 = (_BYTE *)*v51;
      if ( v97 == (_BYTE *)*v51 )
        break;
    }
  }
  v53 = v51[1];
  if ( v36 == **(_QWORD **)(v53 + 32) )
    goto LABEL_41;
  v54 = v49 & (((unsigned int)v36 >> 9) ^ ((unsigned int)v36 >> 4));
  v55 = (__int64 *)(v48 + 16LL * v54);
  v56 = *v55;
  if ( v36 != *v55 )
  {
    v85 = 1;
    while ( v56 != -4096 )
    {
      v86 = v85 + 1;
      v87 = v49 & (v54 + v85);
      v54 = v87;
      v55 = (__int64 *)(v48 + 16 * v87);
      v56 = *v55;
      if ( v36 == *v55 )
        goto LABEL_67;
      v85 = v86;
    }
    goto LABEL_41;
  }
LABEL_67:
  if ( v53 != v55[1] )
    goto LABEL_41;
  sub_276A3A0(&v116, v90, v36, v95, v18, (unsigned int)(v94 + 1));
  if ( v116.m128i_i64[1] == v116.m128i_i64[0] )
  {
LABEL_107:
    sub_27677F0((unsigned __int64 *)&v116);
    goto LABEL_41;
  }
  v88 = v31;
  v57 = v7;
  v58 = (unsigned __int64 *)v116.m128i_i64[0];
  v59 = v18;
  v60 = v116.m128i_i64[1];
  while ( 1 )
  {
    v63 = v58[2];
    if ( v63 == v58[3] )
    {
      sub_2769700(v58, &v97);
    }
    else
    {
      *(_QWORD *)(v63 - 8) = v97;
      v58[2] -= 8LL;
    }
    v61 = *(__int64 **)(v57 + 8);
    if ( v61 == *(__int64 **)(v57 + 16) )
    {
      sub_276A200(v57, *(const void **)(v57 + 8), v58);
      v62 = *(__int64 **)(v57 + 8);
    }
    else
    {
      if ( v61 )
      {
        sub_276A0C0(v61, v58);
        v61 = *(__int64 **)(v57 + 8);
      }
      v62 = v61 + 10;
      *(_QWORD *)(v57 + 8) = v62;
    }
    if ( (unsigned int)qword_4FFAD28 <= 0xCCCCCCCCCCCCCCCDLL * (((__int64)v62 - *(_QWORD *)v57) >> 4) )
      break;
    v58 += 10;
    if ( (unsigned __int64 *)v60 == v58 )
    {
      v7 = v57;
      v31 = v88;
      v18 = v59;
      goto LABEL_107;
    }
  }
  v7 = v57;
  sub_27677F0((unsigned __int64 *)&v116);
LABEL_52:
  if ( !v108.m128i_i8[4] )
    _libc_free((unsigned __int64)v106);
  return v7;
}
