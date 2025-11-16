// Function: sub_195F380
// Address: 0x195f380
//
__int64 __fastcall sub_195F380(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        _QWORD *a7,
        __int64 a8,
        __int64 a9,
        char *a10,
        __int64 *a11)
{
  __int64 v12; // rax
  __int64 *v13; // rax
  __int64 v14; // r13
  __int64 v15; // rbx
  _QWORD *v16; // rax
  __int64 v17; // r14
  char v18; // al
  unsigned __int16 v19; // ax
  unsigned int v20; // r15d
  int v22; // r8d
  int v23; // r9d
  __int64 v24; // rax
  __int64 v25; // rax
  __m128i v26; // xmm0
  __m128i v27; // xmm1
  char *v28; // r12
  _QWORD *v29; // r12
  _QWORD *v30; // r13
  _QWORD *v31; // rdi
  unsigned __int8 *v32; // rsi
  __m128i v33; // xmm2
  __int64 v34; // rdx
  unsigned __int64 v35; // rbx
  _QWORD *v36; // rax
  __int64 v37; // r14
  __int16 v38; // ax
  unsigned __int8 **v39; // rbx
  __int64 v40; // rax
  unsigned __int16 v41; // ax
  unsigned int v42; // r12d
  __int64 v43; // rbx
  char v44; // al
  unsigned __int64 v45; // rsi
  __int64 *v46; // r12
  __int64 v47; // rax
  __int64 *v48; // rsi
  __int64 v49; // rax
  __int64 *v50; // rbx
  __int64 v51; // rax
  __int64 v52; // r13
  __int64 v53; // r14
  __int64 v54; // rax
  unsigned __int64 v55; // rax
  __int64 v56; // rax
  signed __int64 v57; // rax
  __int64 v58; // rax
  __int64 v59; // rax
  __int64 v60; // rax
  char *v61; // r13
  char *v62; // rdi
  __int64 v63; // rsi
  unsigned __int8 *v64; // rsi
  __m128i v65; // xmm3
  __int64 v66; // rax
  __int64 v67; // rax
  __int64 v68; // r12
  __int64 *v70; // [rsp+18h] [rbp-708h]
  bool v75; // [rsp+40h] [rbp-6E0h]
  __int64 v76; // [rsp+40h] [rbp-6E0h]
  __int64 v77; // [rsp+48h] [rbp-6D8h]
  __int64 v78; // [rsp+50h] [rbp-6D0h]
  unsigned int v79; // [rsp+58h] [rbp-6C8h]
  char v80; // [rsp+5Ch] [rbp-6C4h]
  char v81; // [rsp+5Dh] [rbp-6C3h]
  char v82; // [rsp+5Eh] [rbp-6C2h]
  unsigned __int8 v83; // [rsp+5Fh] [rbp-6C1h]
  unsigned __int8 v84; // [rsp+5Fh] [rbp-6C1h]
  __int64 v85; // [rsp+60h] [rbp-6C0h]
  __int64 *v86; // [rsp+68h] [rbp-6B8h]
  _QWORD *v87; // [rsp+78h] [rbp-6A8h]
  unsigned __int8 *v88; // [rsp+88h] [rbp-698h] BYREF
  _QWORD v89[2]; // [rsp+90h] [rbp-690h] BYREF
  __m128i v90; // [rsp+A0h] [rbp-680h] BYREF
  __int64 v91; // [rsp+B0h] [rbp-670h]
  unsigned __int8 *v92[2]; // [rsp+C0h] [rbp-660h] BYREF
  __int16 v93; // [rsp+D0h] [rbp-650h]
  _BYTE v94[64]; // [rsp+E0h] [rbp-640h] BYREF
  __int64 (__fastcall **v95)(); // [rsp+120h] [rbp-600h] BYREF
  int v96; // [rsp+128h] [rbp-5F8h]
  char v97; // [rsp+12Ch] [rbp-5F4h]
  _QWORD *v98; // [rsp+130h] [rbp-5F0h]
  __m128i v99; // [rsp+138h] [rbp-5E8h]
  __int64 v100; // [rsp+148h] [rbp-5D8h]
  __int64 v101; // [rsp+150h] [rbp-5D0h]
  __m128i v102; // [rsp+158h] [rbp-5C8h]
  unsigned __int8 *v103; // [rsp+168h] [rbp-5B8h] BYREF
  unsigned int v104; // [rsp+170h] [rbp-5B0h]
  char v105; // [rsp+174h] [rbp-5ACh]
  __m128i v106; // [rsp+178h] [rbp-5A8h] BYREF
  _QWORD v107[44]; // [rsp+188h] [rbp-598h] BYREF
  char v108; // [rsp+2E8h] [rbp-438h]
  int v109; // [rsp+2ECh] [rbp-434h]
  __int64 v110; // [rsp+2F0h] [rbp-430h]
  _QWORD *v111; // [rsp+300h] [rbp-420h] BYREF
  __int64 v112; // [rsp+308h] [rbp-418h]
  _QWORD *v113; // [rsp+310h] [rbp-410h] BYREF
  __m128i v114; // [rsp+318h] [rbp-408h] BYREF
  __int64 v115; // [rsp+328h] [rbp-3F8h]
  __int64 v116; // [rsp+330h] [rbp-3F0h]
  __m128i v117; // [rsp+338h] [rbp-3E8h] BYREF
  unsigned __int8 *v118; // [rsp+348h] [rbp-3D8h]
  char v119; // [rsp+350h] [rbp-3D0h]
  char *v120; // [rsp+358h] [rbp-3C8h] BYREF
  int v121; // [rsp+360h] [rbp-3C0h]
  char v122; // [rsp+368h] [rbp-3B8h] BYREF
  char v123; // [rsp+4C8h] [rbp-258h]
  int v124; // [rsp+4CCh] [rbp-254h]
  __int64 v125; // [rsp+4D0h] [rbp-250h]
  __int64 *v126; // [rsp+4E0h] [rbp-240h] BYREF
  __int64 v127; // [rsp+4E8h] [rbp-238h]
  _BYTE v128[560]; // [rsp+4F0h] [rbp-230h] BYREF

  v87 = **(_QWORD ***)(a1 + 80);
  v127 = 0x4000000000LL;
  v126 = (__int64 *)v128;
  v77 = sub_13FC520(a8);
  v90 = 0u;
  v91 = 0;
  v12 = sub_157EB90(v77);
  v78 = sub_1632FA0(v12);
  v80 = *a10;
  if ( *a10 )
  {
    v43 = sub_14AD280((__int64)v87, v78, 6u);
    v44 = *(_BYTE *)(v43 + 16);
    if ( v44 != 53 )
    {
      if ( !(unsigned __int8)sub_140B1C0(v43, a7, 0) || (unsigned __int8)sub_139D0F0(v43, 1) )
        goto LABEL_12;
      v44 = *(_BYTE *)(v43 + 16);
    }
    v80 = v44 != 53;
  }
  v13 = *(__int64 **)(a1 + 80);
  v70 = &v13[*(unsigned int *)(a1 + 88)];
  if ( v13 == v70 )
    goto LABEL_12;
  v86 = *(__int64 **)(a1 + 80);
  v79 = 1;
  v81 = 0;
  v14 = a8 + 56;
  v82 = 0;
  v75 = 0;
  v83 = 0;
  do
  {
    if ( *v87 != *(_QWORD *)*v86 )
      goto LABEL_12;
    v15 = *(_QWORD *)(*v86 + 8);
    if ( v15 )
    {
      v85 = *v86;
      while ( 1 )
      {
        v16 = sub_1648700(v15);
        v17 = (__int64)v16;
        if ( *((_BYTE *)v16 + 16) > 0x17u && sub_1377F70(v14, v16[5]) )
        {
          v18 = *(_BYTE *)(v17 + 16);
          if ( v18 == 54 )
          {
            v19 = *(_WORD *)(v17 + 18);
            if ( ((v19 >> 7) & 6) != 0 || (v19 & 1) != 0 )
              goto LABEL_12;
            v82 |= sub_15F32D0(v17);
            v81 |= !sub_15F32D0(v17);
            if ( !v83 )
            {
              v45 = sub_157EBA0(v77);
              v83 = sub_14AF470(v17, v45, a6, 0);
              if ( !v83 )
                v83 = sub_195F310(v17, a6, a8, a10, a11);
            }
            v24 = (unsigned int)v127;
            if ( !(_DWORD)v127 )
              goto LABEL_83;
            goto LABEL_18;
          }
          if ( v18 != 55 )
            goto LABEL_12;
          v40 = (*(_BYTE *)(v17 + 23) & 0x40) != 0
              ? *(_QWORD *)(v17 - 8)
              : v17 - 24LL * (*(_DWORD *)(v17 + 20) & 0xFFFFFFF);
          if ( v85 == *(_QWORD *)(v40 + 24) )
            break;
        }
LABEL_23:
        v15 = *(_QWORD *)(v15 + 8);
        if ( !v15 )
          goto LABEL_24;
      }
      v41 = *(_WORD *)(v17 + 18);
      if ( ((v41 >> 7) & 6) != 0 || (v41 & 1) != 0 )
        goto LABEL_12;
      v82 |= sub_15F32D0(v17);
      v81 |= !sub_15F32D0(v17);
      v42 = 1 << (*(unsigned __int16 *)(v17 + 18) >> 1) >> 1;
      if ( !v42 )
        v42 = sub_15A9FE0(v78, **(_QWORD **)(v17 - 48));
      v22 = v83;
      LOBYTE(v22) = v75 & v83;
      if ( (v75 & v83) != 0 )
      {
        if ( v42 <= v79 )
          goto LABEL_81;
        v84 = v75 & v83;
        if ( !(unsigned __int8)sub_1437020(v17, a6, a8, a10) )
        {
          v22 = v84;
LABEL_81:
          v75 = v22;
LABEL_82:
          v24 = (unsigned int)v127;
          v83 = v22;
          if ( !(_DWORD)v127 )
          {
LABEL_83:
            sub_14A8180(v17, v90.m128i_i64, 0);
            v24 = (unsigned int)v127;
LABEL_20:
            if ( HIDWORD(v127) <= (unsigned int)v24 )
            {
              sub_16CD150((__int64)&v126, v128, 0, 8, v22, v23);
              v24 = (unsigned int)v127;
            }
            v126[v24] = v17;
            LODWORD(v127) = v127 + 1;
            goto LABEL_23;
          }
LABEL_18:
          if ( v90.m128i_i64[0] || __PAIR128__(v90.m128i_u64[1], 0) != v91 )
          {
            sub_14A8180(v17, v90.m128i_i64, 1);
            v24 = (unsigned int)v127;
          }
          goto LABEL_20;
        }
        goto LABEL_108;
      }
      if ( (unsigned __int8)sub_1437020(v17, a6, a8, a10) )
      {
        if ( v42 <= v79 )
        {
LABEL_107:
          v75 = 1;
          v22 = 1;
          goto LABEL_82;
        }
LABEL_108:
        v79 = v42;
        goto LABEL_107;
      }
      if ( v75 )
      {
LABEL_105:
        v55 = sub_157EBA0(v77);
        v22 = sub_13F8190(*(_QWORD *)(v17 - 24), 1 << (*(unsigned __int16 *)(v17 + 18) >> 1) >> 1, v78, v55, a6);
        goto LABEL_82;
      }
      v46 = *(__int64 **)a2;
      v47 = 8LL * *(unsigned int *)(a2 + 8);
      v48 = (__int64 *)(*(_QWORD *)a2 + v47);
      v49 = v47 >> 5;
      if ( v49 )
      {
        v76 = v15;
        v50 = &v46[4 * v49];
        v51 = v14;
        v52 = v17;
        v53 = v51;
        while ( 1 )
        {
          if ( !sub_15CC8F0(a6, *(_QWORD *)(v52 + 40), *v46) )
          {
            v54 = v53;
            v17 = v52;
            v15 = v76;
            v14 = v54;
            v75 = v48 == v46;
            goto LABEL_104;
          }
          if ( !sub_15CC8F0(a6, *(_QWORD *)(v52 + 40), v46[1]) )
            break;
          if ( !sub_15CC8F0(a6, *(_QWORD *)(v52 + 40), v46[2]) )
          {
            v59 = v53;
            v17 = v52;
            v15 = v76;
            v14 = v59;
            v75 = v48 == v46 + 2;
            goto LABEL_104;
          }
          if ( !sub_15CC8F0(a6, *(_QWORD *)(v52 + 40), v46[3]) )
          {
            v58 = v53;
            v17 = v52;
            v15 = v76;
            v14 = v58;
            v75 = v48 == v46 + 3;
            goto LABEL_104;
          }
          v46 += 4;
          if ( v50 == v46 )
          {
            v56 = v53;
            v15 = v76;
            v17 = v52;
            v14 = v56;
            goto LABEL_110;
          }
        }
        v60 = v53;
        v17 = v52;
        v15 = v76;
        v14 = v60;
        v75 = v48 == v46 + 1;
LABEL_104:
        if ( v83 )
        {
          v22 = v83;
          goto LABEL_82;
        }
        goto LABEL_105;
      }
LABEL_110:
      v57 = (char *)v48 - (char *)v46;
      if ( (char *)v48 - (char *)v46 != 16 )
      {
        if ( v57 != 24 )
        {
          if ( v57 != 8 )
          {
            v75 = 1;
            goto LABEL_104;
          }
LABEL_128:
          v75 = sub_15CC8F0(a6, *(_QWORD *)(v17 + 40), *v46);
          if ( v75 )
            goto LABEL_104;
          goto LABEL_129;
        }
        if ( !sub_15CC8F0(a6, *(_QWORD *)(v17 + 40), *v46) )
        {
LABEL_129:
          v75 = v48 == v46;
          goto LABEL_104;
        }
        ++v46;
      }
      if ( sub_15CC8F0(a6, *(_QWORD *)(v17 + 40), *v46) )
      {
        ++v46;
        goto LABEL_128;
      }
      goto LABEL_129;
    }
LABEL_24:
    ++v86;
  }
  while ( v70 != v86 );
  v20 = v83;
  LOBYTE(v20) = (v82 & v81 ^ 1) & v83;
  if ( !(_BYTE)v20
    || !v80
    && !v75
    && ((v68 = sub_14AD280((__int64)v87, v78, 6u), !(unsigned __int8)sub_140B1C0(v68, a7, 0))
     && *(_BYTE *)(v68 + 16) != 53
     || (unsigned __int8)sub_139D0F0(v68, 1)) )
  {
LABEL_12:
    v20 = 0;
    goto LABEL_13;
  }
  v25 = sub_15E0530(*a11);
  if ( !sub_1602790(v25) )
  {
    v66 = sub_15E0530(*a11);
    v67 = sub_16033E0(v66);
    if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v67 + 48LL))(v67) )
      goto LABEL_45;
  }
  sub_15CA3B0((__int64)&v111, (__int64)"licm", (__int64)"PromoteLoopAccessesToScalar", 27, *v126);
  sub_15CAB20((__int64)&v111, "Moving accesses to memory location out of the loop", 0x32u);
  v26 = _mm_loadu_si128(&v114);
  v27 = _mm_loadu_si128(&v117);
  v96 = v112;
  v99 = v26;
  v97 = BYTE4(v112);
  v102 = v27;
  v98 = v113;
  v100 = v115;
  v95 = (__int64 (__fastcall **)())&unk_49ECF68;
  v101 = v116;
  LOBYTE(v104) = v119;
  if ( v119 )
    v103 = v118;
  v106.m128i_i64[0] = (__int64)v107;
  v106.m128i_i64[1] = 0x400000000LL;
  if ( !v121 )
  {
    v108 = v123;
    v109 = v124;
    v110 = v125;
    v95 = (__int64 (__fastcall **)())&unk_49ECF98;
    goto LABEL_33;
  }
  sub_195ED40((__int64)&v106, (__int64)&v120);
  v28 = v120;
  v108 = v123;
  v109 = v124;
  v110 = v125;
  v95 = (__int64 (__fastcall **)())&unk_49ECF98;
  v111 = &unk_49ECF68;
  v61 = &v120[88 * v121];
  if ( v120 != v61 )
  {
    do
    {
      v61 -= 88;
      v62 = (char *)*((_QWORD *)v61 + 4);
      if ( v62 != v61 + 48 )
        j_j___libc_free_0(v62, *((_QWORD *)v61 + 6) + 1LL);
      if ( *(char **)v61 != v61 + 16 )
        j_j___libc_free_0(*(_QWORD *)v61, *((_QWORD *)v61 + 2) + 1LL);
    }
    while ( v28 != v61 );
LABEL_33:
    v28 = v120;
  }
  if ( v28 != &v122 )
    _libc_free((unsigned __int64)v28);
  sub_143AA50(a11, (__int64)&v95);
  v29 = (_QWORD *)v106.m128i_i64[0];
  v95 = (__int64 (__fastcall **)())&unk_49ECF68;
  v30 = (_QWORD *)(v106.m128i_i64[0] + 88LL * v106.m128i_u32[2]);
  if ( (_QWORD *)v106.m128i_i64[0] != v30 )
  {
    do
    {
      v30 -= 11;
      v31 = (_QWORD *)v30[4];
      if ( v31 != v30 + 6 )
        j_j___libc_free_0(v31, v30[6] + 1LL);
      if ( (_QWORD *)*v30 != v30 + 2 )
        j_j___libc_free_0(*v30, v30[2] + 1LL);
    }
    while ( v29 != v30 );
    v29 = (_QWORD *)v106.m128i_i64[0];
  }
  if ( v29 != v107 )
    _libc_free((unsigned __int64)v29);
LABEL_45:
  v32 = *(unsigned __int8 **)(*v126 + 48);
  v88 = v32;
  if ( v32 )
    sub_1623A60((__int64)&v88, (__int64)v32, 2);
  v111 = &v113;
  v112 = 0x1000000000LL;
  sub_1B3B830(v94, &v111);
  v92[0] = v88;
  if ( v88 )
    sub_1623A60((__int64)v92, (__int64)v88, 2);
  sub_1B3BD80(&v95, v126, (unsigned int)v127, v94, 0, 0);
  v99.m128i_i64[0] = a1;
  v95 = off_49F3B68;
  v103 = v92[0];
  v98 = v87;
  v99.m128i_i64[1] = a2;
  v100 = a3;
  v101 = a4;
  v102.m128i_i64[0] = a9;
  v102.m128i_i64[1] = a5;
  if ( v92[0] )
  {
    sub_1623210((__int64)v92, v92[0], (__int64)&v103);
    v33 = _mm_loadu_si128(&v90);
    v104 = v79;
    v106 = v33;
  }
  else
  {
    v65 = _mm_loadu_si128(&v90);
    v104 = v79;
    v106 = v65;
  }
  v105 = v82;
  v107[0] = v91;
  v89[0] = sub_1649960((__int64)v87);
  v93 = 773;
  v92[0] = (unsigned __int8 *)v89;
  v89[1] = v34;
  v92[1] = (unsigned __int8 *)".promoted";
  v35 = sub_157EBA0(v77);
  v36 = sub_1648A60(64, 1u);
  v37 = (__int64)v36;
  if ( v36 )
    sub_15F90E0((__int64)v36, (__int64)v87, (__int64)v92, v35);
  if ( v82 )
  {
    v38 = *(_WORD *)(v37 + 18) & 0xFC7F;
    LOBYTE(v38) = v38 | 0x80;
    *(_WORD *)(v37 + 18) = v38;
  }
  v39 = (unsigned __int8 **)(v37 + 48);
  sub_15F8F50(v37, v79);
  v92[0] = v88;
  if ( !v88 )
  {
    if ( v39 == v92 )
      goto LABEL_59;
    v63 = *(_QWORD *)(v37 + 48);
    if ( !v63 )
      goto LABEL_59;
LABEL_132:
    sub_161E7C0(v37 + 48, v63);
    goto LABEL_133;
  }
  sub_1623A60((__int64)v92, (__int64)v88, 2);
  if ( v39 == v92 )
  {
    if ( v92[0] )
      sub_161E7C0((__int64)v92, (__int64)v92[0]);
    goto LABEL_59;
  }
  v63 = *(_QWORD *)(v37 + 48);
  if ( v63 )
    goto LABEL_132;
LABEL_133:
  v64 = v92[0];
  *(unsigned __int8 **)(v37 + 48) = v92[0];
  if ( v64 )
    sub_1623210((__int64)v92, v64, v37 + 48);
LABEL_59:
  if ( v90.m128i_i64[0] || __PAIR128__(v90.m128i_u64[1], 0) != v91 )
    sub_1626170(v37, v90.m128i_i64);
  sub_1B3BE00(v94, v77, v37);
  sub_1B40B80(&v95, &v126);
  if ( !*(_QWORD *)(v37 + 8) )
    sub_15F20C0((_QWORD *)v37);
  v95 = off_49F3B68;
  if ( v103 )
    sub_161E7C0((__int64)&v103, (__int64)v103);
  sub_1B3B860(v94);
  if ( v111 != &v113 )
    _libc_free((unsigned __int64)v111);
  if ( v88 )
    sub_161E7C0((__int64)&v88, (__int64)v88);
LABEL_13:
  if ( v126 != (__int64 *)v128 )
    _libc_free((unsigned __int64)v126);
  return v20;
}
