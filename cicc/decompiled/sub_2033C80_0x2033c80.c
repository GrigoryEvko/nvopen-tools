// Function: sub_2033C80
// Address: 0x2033c80
//
__int64 *__fastcall sub_2033C80(__int64 *a1, __int64 a2, double a3, __m128i a4, __m128i a5)
{
  const __m128i *v6; // rax
  __int64 v7; // rsi
  __m128 v8; // xmm0
  __int64 v9; // rax
  unsigned __int8 v10; // dl
  __int64 v11; // r8
  __int64 v12; // rsi
  unsigned int v13; // eax
  __int64 *v14; // r15
  const void **v15; // rdx
  const void **v16; // r13
  __int64 v17; // rax
  unsigned int v18; // edx
  unsigned __int8 v19; // al
  __int128 v20; // rax
  unsigned int v21; // edx
  unsigned int v22; // r15d
  __int64 v23; // rax
  __int64 v24; // r8
  unsigned int *v25; // r9
  __int64 v26; // rdx
  unsigned int v27; // edx
  __int64 v28; // r13
  __int64 v29; // rax
  unsigned __int8 v30; // r13
  __int64 v31; // rax
  __int64 (__fastcall *v32)(unsigned int *, __int64, __int64, _QWORD, const void **); // r15
  __int64 v33; // rax
  char v34; // al
  const void **v35; // rdx
  __int64 *v36; // r15
  __int64 v37; // rax
  __int64 v38; // rsi
  __int64 v39; // r12
  __int64 v40; // rdx
  __int64 v41; // r13
  __int64 v42; // rax
  unsigned __int8 v43; // cl
  const void **v44; // r8
  __int32 v45; // eax
  unsigned int v46; // ebx
  __int64 v47; // r10
  __int64 v48; // r11
  __int64 v49; // rdx
  char v50; // al
  __int64 v51; // rdx
  unsigned int v52; // esi
  __int64 *v53; // r12
  bool v55; // al
  unsigned int v56; // r15d
  unsigned int v57; // eax
  __int64 v58; // rsi
  __int64 *v59; // r13
  unsigned int v60; // edx
  unsigned int v61; // edx
  __int64 v62; // rcx
  __int64 *v63; // r15
  __int64 v64; // rax
  __int128 v65; // rax
  unsigned int v66; // r10d
  unsigned int v67; // edx
  unsigned int v68; // r15d
  __int64 v69; // rsi
  __int128 v70; // rax
  __int64 *v71; // rax
  unsigned int v72; // edx
  __int64 v73; // rax
  __int8 v74; // dl
  __int64 v75; // rax
  __m128i v76; // rax
  unsigned int *v77; // r15
  bool v78; // dl
  bool v79; // cl
  bool v80; // al
  bool v81; // al
  bool v82; // al
  __int128 v83; // [rsp-20h] [rbp-120h]
  __int64 v84; // [rsp+10h] [rbp-F0h]
  const void **v85; // [rsp+10h] [rbp-F0h]
  __int128 v86; // [rsp+10h] [rbp-F0h]
  __int128 v87; // [rsp+10h] [rbp-F0h]
  unsigned int *v88; // [rsp+10h] [rbp-F0h]
  unsigned int v89; // [rsp+10h] [rbp-F0h]
  unsigned int *v90; // [rsp+20h] [rbp-E0h]
  __int64 v91; // [rsp+28h] [rbp-D8h]
  __int64 v92; // [rsp+30h] [rbp-D0h]
  __int64 (__fastcall *v93)(__int64, __int64); // [rsp+40h] [rbp-C0h]
  __int64 v94; // [rsp+40h] [rbp-C0h]
  __int64 v95; // [rsp+50h] [rbp-B0h]
  const void **v96; // [rsp+50h] [rbp-B0h]
  unsigned __int8 v97; // [rsp+50h] [rbp-B0h]
  unsigned int *v98; // [rsp+50h] [rbp-B0h]
  unsigned int *v99; // [rsp+50h] [rbp-B0h]
  bool v100; // [rsp+50h] [rbp-B0h]
  bool v101; // [rsp+50h] [rbp-B0h]
  unsigned int v102; // [rsp+58h] [rbp-A8h]
  __int64 v103; // [rsp+58h] [rbp-A8h]
  const void **v104; // [rsp+58h] [rbp-A8h]
  unsigned __int64 v105; // [rsp+60h] [rbp-A0h]
  unsigned __int64 v106; // [rsp+68h] [rbp-98h]
  __int16 *v107; // [rsp+68h] [rbp-98h]
  char v108[8]; // [rsp+70h] [rbp-90h] BYREF
  __int64 v109; // [rsp+78h] [rbp-88h]
  __int64 v110; // [rsp+80h] [rbp-80h] BYREF
  int v111; // [rsp+88h] [rbp-78h]
  unsigned int v112; // [rsp+90h] [rbp-70h] BYREF
  const void **v113; // [rsp+98h] [rbp-68h]
  __m128i v114; // [rsp+A0h] [rbp-60h] BYREF
  __m128i v115; // [rsp+B0h] [rbp-50h] BYREF

  v6 = *(const __m128i **)(a2 + 32);
  v7 = *(_QWORD *)(a2 + 72);
  v8 = (__m128)_mm_loadu_si128(v6);
  v106 = v8.m128_u64[1];
  v9 = *(_QWORD *)(v6->m128i_i64[0] + 40) + 16LL * v6->m128i_u32[2];
  v10 = *(_BYTE *)v9;
  v11 = *(_QWORD *)(v9 + 8);
  v110 = v7;
  v108[0] = v10;
  v109 = v11;
  if ( v7 )
  {
    sub_1623A60((__int64)&v110, v7, 2);
    v11 = v109;
    v10 = v108[0];
  }
  v12 = *a1;
  v111 = *(_DWORD *)(a2 + 64);
  sub_1F40D10((__int64)&v115, v12, *(_QWORD *)(a1[1] + 48), v10, v11);
  if ( v115.m128i_i8[0] == 5 )
  {
    v103 = sub_2032580((__int64)a1, v8.m128_u64[0], v8.m128_i64[1]);
    v22 = v61;
  }
  else
  {
    LOBYTE(v13) = sub_1F7E0F0((__int64)v108);
    v14 = (__int64 *)a1[1];
    v102 = v13;
    v16 = v15;
    v95 = *a1;
    v93 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)*a1 + 48LL);
    v17 = sub_1E0A0C0(v14[4]);
    if ( v93 == sub_1D13A20 )
    {
      v18 = 8 * sub_15A9520(v17, 0);
      if ( v18 == 32 )
      {
        v19 = 5;
      }
      else if ( v18 > 0x20 )
      {
        v19 = 6;
        if ( v18 != 64 )
        {
          v19 = 0;
          if ( v18 == 128 )
            v19 = 7;
        }
      }
      else
      {
        v19 = 3;
        if ( v18 != 8 )
          v19 = 4 * (v18 == 16);
      }
    }
    else
    {
      v19 = v93(v95, v17);
    }
    *(_QWORD *)&v20 = sub_1D38BB0(
                        (__int64)v14,
                        0,
                        (__int64)&v110,
                        v19,
                        0,
                        0,
                        (__m128i)v8,
                        *(double *)a4.m128i_i64,
                        a5,
                        0);
    v103 = (__int64)sub_1D332F0(
                      v14,
                      106,
                      (__int64)&v110,
                      v102,
                      v16,
                      0,
                      *(double *)v8.m128_u64,
                      *(double *)a4.m128i_i64,
                      a5,
                      v8.m128_i64[0],
                      v8.m128_u64[1],
                      v20);
    v22 = v21;
  }
  v23 = sub_2032580((__int64)a1, *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 48LL));
  v25 = (unsigned int *)*a1;
  v92 = v23;
  v91 = v26;
  v27 = *(_DWORD *)(*a1 + 60);
  v94 = v22;
  v28 = 16LL * v22;
  if ( *(_DWORD *)(*a1 + 64) == v27 )
  {
    v62 = v25[17];
    goto LABEL_42;
  }
  if ( *(_WORD *)(v103 + 24) != 137 )
  {
    v29 = v28 + *(_QWORD *)(v103 + 40);
    v30 = *(_BYTE *)v29;
    v96 = *(const void ***)(v29 + 8);
    goto LABEL_13;
  }
  v73 = *(_QWORD *)(**(_QWORD **)(v103 + 32) + 40LL) + 16LL * *(unsigned int *)(*(_QWORD *)(v103 + 32) + 8LL);
  v74 = *(_BYTE *)v73;
  v75 = *(_QWORD *)(v73 + 8);
  v114.m128i_i8[0] = v74;
  v114.m128i_i64[1] = v75;
  if ( v74 )
  {
    if ( (unsigned __int8)(v74 - 14) > 0x5Fu )
    {
LABEL_61:
      v76 = v114;
      v77 = v25;
      goto LABEL_62;
    }
  }
  else
  {
    v98 = v25;
    v80 = sub_1F58D20((__int64)&v114);
    v25 = v98;
    if ( !v80 )
      goto LABEL_61;
  }
  v99 = v25;
  v76.m128i_i8[0] = sub_1F7E0F0((__int64)&v114);
  v77 = (unsigned int *)*a1;
  v25 = v99;
LABEL_62:
  v115.m128i_i8[0] = v76.m128i_i8[0];
  v115.m128i_i64[1] = v76.m128i_i64[1];
  if ( v76.m128i_i8[0] )
  {
    if ( (unsigned __int8)(v76.m128i_i8[0] - 14) > 0x5Fu )
    {
      v78 = (unsigned __int8)(v76.m128i_i8[0] - 86) <= 0x17u || (unsigned __int8)(v76.m128i_i8[0] - 8) <= 5u;
      goto LABEL_65;
    }
LABEL_76:
    v27 = v25[17];
    goto LABEL_67;
  }
  v88 = v25;
  v100 = sub_1F58CD0((__int64)&v115);
  v81 = sub_1F58D20((__int64)&v115);
  v78 = v100;
  v25 = v88;
  if ( v81 )
    goto LABEL_76;
LABEL_65:
  if ( v78 )
    v27 = v25[16];
  else
    v27 = v25[15];
LABEL_67:
  a4 = _mm_loadu_si128(&v114);
  v115 = a4;
  if ( v114.m128i_i8[0] )
  {
    if ( (unsigned __int8)(v114.m128i_i8[0] - 14) > 0x5Fu )
    {
      v79 = (unsigned __int8)(v114.m128i_i8[0] - 86) <= 0x17u || (unsigned __int8)(v114.m128i_i8[0] - 8) <= 5u;
      goto LABEL_70;
    }
LABEL_79:
    v62 = v77[17];
    goto LABEL_42;
  }
  v89 = v27;
  v101 = sub_1F58CD0((__int64)&v115);
  v82 = sub_1F58D20((__int64)&v115);
  v79 = v101;
  v27 = v89;
  if ( v82 )
    goto LABEL_79;
LABEL_70:
  if ( v79 )
    v62 = v77[16];
  else
    v62 = v77[15];
LABEL_42:
  v63 = (__int64 *)a1[1];
  v64 = v28 + *(_QWORD *)(v103 + 40);
  v30 = *(_BYTE *)v64;
  v96 = *(const void ***)(v64 + 8);
  if ( v27 != (_DWORD)v62 )
  {
    if ( v27 == 1 )
    {
      v115.m128i_i64[0] = *(_QWORD *)(a2 + 72);
      if ( v115.m128i_i64[0] )
        sub_20219D0(v115.m128i_i64);
      v115.m128i_i32[2] = *(_DWORD *)(a2 + 64);
      *(_QWORD *)&v65 = sub_1D38BB0(
                          (__int64)v63,
                          1,
                          (__int64)&v115,
                          v30,
                          v96,
                          0,
                          (__m128i)v8,
                          *(double *)a4.m128i_i64,
                          a5,
                          0);
      v66 = v30;
      v114.m128i_i64[0] = *(_QWORD *)(a2 + 72);
      if ( v114.m128i_i64[0] )
      {
        v86 = v65;
        sub_20219D0(v114.m128i_i64);
        v66 = v30;
        v65 = v86;
      }
      v114.m128i_i32[2] = *(_DWORD *)(a2 + 64);
      v106 = v94 | v8.m128_u64[1] & 0xFFFFFFFF00000000LL;
      v103 = (__int64)sub_1D332F0(
                        v63,
                        118,
                        (__int64)&v114,
                        v66,
                        v96,
                        0,
                        *(double *)v8.m128_u64,
                        *(double *)a4.m128i_i64,
                        a5,
                        v103,
                        v106,
                        v65);
      v68 = v67;
      if ( v114.m128i_i64[0] )
        sub_161E7C0((__int64)&v114, v114.m128i_i64[0]);
      v69 = v115.m128i_i64[0];
      if ( !v115.m128i_i64[0] )
        goto LABEL_52;
      goto LABEL_51;
    }
    if ( v27 == 2 )
    {
      *(_QWORD *)&v70 = sub_1D2EF30(v63, 2, 0, v62, v24, (__int64)v25);
      v115.m128i_i64[0] = *(_QWORD *)(a2 + 72);
      if ( v115.m128i_i64[0] )
      {
        v87 = v70;
        sub_20219D0(v115.m128i_i64);
        v70 = v87;
      }
      v115.m128i_i32[2] = *(_DWORD *)(a2 + 64);
      v106 = v94 | v8.m128_u64[1] & 0xFFFFFFFF00000000LL;
      v71 = sub_1D332F0(
              v63,
              148,
              (__int64)&v115,
              v30,
              v96,
              0,
              *(double *)v8.m128_u64,
              *(double *)a4.m128i_i64,
              a5,
              v103,
              v106,
              v70);
      v69 = v115.m128i_i64[0];
      v103 = (__int64)v71;
      v68 = v72;
      if ( !v115.m128i_i64[0] )
        goto LABEL_52;
LABEL_51:
      sub_161E7C0((__int64)&v115, v69);
LABEL_52:
      v25 = (unsigned int *)*a1;
      v94 = v68;
      goto LABEL_13;
    }
  }
  v25 = (unsigned int *)*a1;
LABEL_13:
  v31 = a1[1];
  v90 = v25;
  v32 = *(__int64 (__fastcall **)(unsigned int *, __int64, __int64, _QWORD, const void **))(*(_QWORD *)v25 + 264LL);
  v84 = *(_QWORD *)(v31 + 48);
  v33 = sub_1E0A0C0(*(_QWORD *)(v31 + 32));
  v34 = v32(v90, v33, v84, v30, v96);
  v115.m128i_i8[0] = v30;
  LOBYTE(v112) = v34;
  v113 = v35;
  v115.m128i_i64[1] = (__int64)v96;
  if ( v30 == v34 )
  {
    if ( v30 || v96 == v35 )
      goto LABEL_15;
LABEL_38:
    v56 = sub_1F58D40((__int64)&v112);
    if ( !v30 )
      goto LABEL_39;
LABEL_31:
    v57 = sub_2021900(v30);
    goto LABEL_32;
  }
  if ( !v34 )
    goto LABEL_38;
  v56 = sub_2021900(v34);
  if ( v30 )
    goto LABEL_31;
LABEL_39:
  v57 = sub_1F58D40((__int64)&v115);
LABEL_32:
  if ( v57 > v56 )
  {
    v58 = *(_QWORD *)(a2 + 72);
    v59 = (__int64 *)a1[1];
    v115.m128i_i64[0] = v58;
    if ( v58 )
      sub_1623A60((__int64)&v115, v58, 2);
    v115.m128i_i32[2] = *(_DWORD *)(a2 + 64);
    v106 = v94 | v106 & 0xFFFFFFFF00000000LL;
    v103 = sub_1D309E0(
             v59,
             145,
             (__int64)&v115,
             v112,
             v113,
             0,
             *(double *)v8.m128_u64,
             *(double *)a4.m128i_i64,
             *(double *)a5.m128i_i64,
             __PAIR128__(v106, v103));
    v94 = v60;
    if ( v115.m128i_i64[0] )
      sub_161E7C0((__int64)&v115, v115.m128i_i64[0]);
  }
LABEL_15:
  v36 = (__int64 *)a1[1];
  v37 = sub_2032580((__int64)a1, *(_QWORD *)(*(_QWORD *)(a2 + 32) + 80LL), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 88LL));
  v38 = *(_QWORD *)(a2 + 72);
  v39 = v37;
  v41 = v40;
  v42 = *(_QWORD *)(v92 + 40) + 16LL * (unsigned int)v91;
  v43 = *(_BYTE *)v42;
  v44 = *(const void ***)(v42 + 8);
  v114.m128i_i64[0] = v38;
  if ( v38 )
  {
    v85 = v44;
    v97 = v43;
    sub_1623A60((__int64)&v114, v38, 2);
    v44 = v85;
    v43 = v97;
  }
  v45 = *(_DWORD *)(a2 + 64);
  v46 = v43;
  v47 = v92;
  v114.m128i_i32[2] = v45;
  v105 = v103;
  v48 = v91;
  v49 = 16 * v94 + *(_QWORD *)(v103 + 40);
  v107 = (__int16 *)(v94 | v106 & 0xFFFFFFFF00000000LL);
  v50 = *(_BYTE *)v49;
  v51 = *(_QWORD *)(v49 + 8);
  v115.m128i_i8[0] = v50;
  v115.m128i_i64[1] = v51;
  if ( v50 )
  {
    v52 = ((unsigned __int8)(v50 - 14) < 0x60u) + 134;
  }
  else
  {
    v104 = v44;
    v55 = sub_1F58D20((__int64)&v115);
    v47 = v92;
    v48 = v91;
    v44 = v104;
    v52 = 134 - (!v55 - 1);
  }
  *((_QWORD *)&v83 + 1) = v48;
  *(_QWORD *)&v83 = v47;
  v53 = sub_1D3A900(v36, v52, (__int64)&v114, v46, v44, 0, v8, *(double *)a4.m128i_i64, a5, v105, v107, v83, v39, v41);
  if ( v114.m128i_i64[0] )
    sub_161E7C0((__int64)&v114, v114.m128i_i64[0]);
  if ( v110 )
    sub_161E7C0((__int64)&v110, v110);
  return v53;
}
