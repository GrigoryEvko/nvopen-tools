// Function: sub_212AC20
// Address: 0x212ac20
//
__int64 __fastcall sub_212AC20(__int64 a1, __int64 a2, __int64 a3, __m128i *a4, __m128i a5, double a6, __m128i a7)
{
  __int64 v9; // rax
  __m128i v10; // xmm1
  __int64 v11; // r15
  __int64 v12; // r13
  __int64 v13; // r13
  char v14; // di
  const void **v15; // r13
  char v16; // r8
  unsigned int v17; // r15d
  const void **v18; // rdx
  unsigned int v19; // r13d
  __int64 v20; // rsi
  unsigned int v21; // eax
  unsigned int v22; // eax
  int v23; // eax
  unsigned int v24; // esi
  __int64 v25; // rax
  __int64 v26; // rdi
  __int64 v27; // r8
  __int64 v28; // r8
  unsigned __int64 v29; // r8
  bool v30; // r15
  unsigned int v31; // r15d
  unsigned int v32; // r14d
  bool v34; // al
  unsigned __int64 *v35; // rax
  __int64 v36; // rdx
  __int64 *v37; // r15
  __int128 v38; // rax
  __int64 v39; // rdx
  __int64 *v40; // r10
  unsigned int v41; // r13d
  unsigned int v42; // r15d
  __int128 v43; // rax
  __int64 *v44; // rax
  unsigned __int64 v45; // rdx
  __int128 v46; // rax
  __int64 v47; // rcx
  const void **v48; // r8
  int v49; // edx
  __int64 *v50; // r12
  __int64 *v51; // rax
  unsigned __int64 v52; // rdx
  __int32 v53; // edx
  __m128i v54; // xmm0
  bool v55; // al
  char v56; // r8
  unsigned int v57; // eax
  unsigned int v58; // eax
  unsigned int v59; // edx
  char v60; // al
  __int64 v61; // rdx
  unsigned int v62; // eax
  __int64 *v63; // r15
  __int64 v64; // rsi
  unsigned __int64 v65; // rdx
  __int128 v66; // rax
  unsigned int v67; // edx
  __int16 v68; // ax
  __int64 v69; // rdi
  __int64 v70; // rax
  __int64 v71; // rcx
  const void **v72; // r8
  int v73; // edx
  __int32 v74; // edx
  __int64 v75; // rax
  __int64 v76; // rcx
  const void **v77; // r8
  __int32 v78; // edx
  int v79; // edx
  __int64 *v80; // r15
  __int128 v81; // rax
  __int64 *v82; // rax
  __int64 v83; // rcx
  const void **v84; // r8
  __int32 v85; // edx
  __int64 *v86; // rax
  int v87; // edx
  __int128 v88; // [rsp+0h] [rbp-1C0h]
  int v89; // [rsp+10h] [rbp-1B0h]
  __int64 *v90; // [rsp+10h] [rbp-1B0h]
  __int128 v91; // [rsp+10h] [rbp-1B0h]
  char v92; // [rsp+10h] [rbp-1B0h]
  char v93; // [rsp+10h] [rbp-1B0h]
  _QWORD *v94; // [rsp+10h] [rbp-1B0h]
  __int128 v97; // [rsp+30h] [rbp-190h]
  __int64 *v98; // [rsp+60h] [rbp-160h]
  unsigned int v99; // [rsp+F0h] [rbp-D0h] BYREF
  const void **v100; // [rsp+F8h] [rbp-C8h]
  unsigned int v101; // [rsp+100h] [rbp-C0h] BYREF
  const void **v102; // [rsp+108h] [rbp-B8h]
  __int64 v103; // [rsp+110h] [rbp-B0h] BYREF
  int v104; // [rsp+118h] [rbp-A8h]
  unsigned __int64 v105; // [rsp+120h] [rbp-A0h] BYREF
  unsigned int v106; // [rsp+128h] [rbp-98h]
  __m128i v107; // [rsp+130h] [rbp-90h] BYREF
  _QWORD *v108; // [rsp+140h] [rbp-80h] BYREF
  unsigned __int64 v109; // [rsp+148h] [rbp-78h]
  _QWORD *v110; // [rsp+150h] [rbp-70h] BYREF
  unsigned int v111; // [rsp+158h] [rbp-68h]
  __int64 v112; // [rsp+160h] [rbp-60h] BYREF
  unsigned int v113; // [rsp+168h] [rbp-58h]
  unsigned __int64 v114; // [rsp+170h] [rbp-50h] BYREF
  __int64 v115; // [rsp+178h] [rbp-48h]
  const void **v116; // [rsp+180h] [rbp-40h] BYREF
  __int64 v117; // [rsp+188h] [rbp-38h]

  v9 = *(_QWORD *)(a2 + 32);
  v10 = _mm_loadu_si128((const __m128i *)(v9 + 40));
  v11 = *(_QWORD *)(v9 + 40);
  v12 = 16LL * *(unsigned int *)(v9 + 48);
  sub_1F40D10(
    (__int64)&v114,
    *(_QWORD *)a1,
    *(_QWORD *)(*(_QWORD *)(a1 + 8) + 48LL),
    **(unsigned __int8 **)(a2 + 40),
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v13 = *(_QWORD *)(v11 + 40) + v12;
  v100 = v116;
  LOBYTE(v99) = v115;
  v14 = *(_BYTE *)v13;
  v15 = *(const void ***)(v13 + 8);
  LOBYTE(v101) = v14;
  v102 = v15;
  if ( v14 )
  {
    if ( (unsigned __int8)(v14 - 14) <= 0x5Fu )
    {
      switch ( v14 )
      {
        case 24:
        case 25:
        case 26:
        case 27:
        case 28:
        case 29:
        case 30:
        case 31:
        case 32:
        case 62:
        case 63:
        case 64:
        case 65:
        case 66:
        case 67:
          v14 = 3;
          break;
        case 33:
        case 34:
        case 35:
        case 36:
        case 37:
        case 38:
        case 39:
        case 40:
        case 68:
        case 69:
        case 70:
        case 71:
        case 72:
        case 73:
          v14 = 4;
          break;
        case 41:
        case 42:
        case 43:
        case 44:
        case 45:
        case 46:
        case 47:
        case 48:
        case 74:
        case 75:
        case 76:
        case 77:
        case 78:
        case 79:
          v14 = 5;
          break;
        case 49:
        case 50:
        case 51:
        case 52:
        case 53:
        case 54:
        case 80:
        case 81:
        case 82:
        case 83:
        case 84:
        case 85:
          v14 = 6;
          break;
        case 55:
          v14 = 7;
          break;
        case 86:
        case 87:
        case 88:
        case 98:
        case 99:
        case 100:
          v14 = 8;
          break;
        case 89:
        case 90:
        case 91:
        case 92:
        case 93:
        case 101:
        case 102:
        case 103:
        case 104:
        case 105:
          v14 = 9;
          break;
        case 94:
        case 95:
        case 96:
        case 97:
        case 106:
        case 107:
        case 108:
        case 109:
          v14 = 10;
          break;
        default:
          v14 = 2;
          break;
      }
    }
    goto LABEL_3;
  }
  v92 = v115;
  v55 = sub_1F58D20((__int64)&v101);
  v56 = v92;
  if ( v55 )
  {
    v60 = sub_1F596B0((__int64)&v101);
    v56 = v99;
    LOBYTE(v114) = v60;
    v14 = v60;
    v115 = v61;
    if ( v60 )
    {
LABEL_3:
      v17 = sub_2127930(v14);
      goto LABEL_4;
    }
  }
  else
  {
    LOBYTE(v114) = 0;
    v115 = (__int64)v15;
  }
  v93 = v56;
  v57 = sub_1F58D40((__int64)&v114);
  v16 = v93;
  v17 = v57;
LABEL_4:
  if ( v16 )
  {
    if ( (unsigned __int8)(v16 - 14) <= 0x5Fu )
    {
      switch ( v16 )
      {
        case 24:
        case 25:
        case 26:
        case 27:
        case 28:
        case 29:
        case 30:
        case 31:
        case 32:
        case 62:
        case 63:
        case 64:
        case 65:
        case 66:
        case 67:
          v16 = 3;
          break;
        case 33:
        case 34:
        case 35:
        case 36:
        case 37:
        case 38:
        case 39:
        case 40:
        case 68:
        case 69:
        case 70:
        case 71:
        case 72:
        case 73:
          v16 = 4;
          break;
        case 41:
        case 42:
        case 43:
        case 44:
        case 45:
        case 46:
        case 47:
        case 48:
        case 74:
        case 75:
        case 76:
        case 77:
        case 78:
        case 79:
          v16 = 5;
          break;
        case 49:
        case 50:
        case 51:
        case 52:
        case 53:
        case 54:
        case 80:
        case 81:
        case 82:
        case 83:
        case 84:
        case 85:
          v16 = 6;
          break;
        case 55:
          v16 = 7;
          break;
        case 86:
        case 87:
        case 88:
        case 98:
        case 99:
        case 100:
          v16 = 8;
          break;
        case 89:
        case 90:
        case 91:
        case 92:
        case 93:
        case 101:
        case 102:
        case 103:
        case 104:
        case 105:
          v16 = 9;
          break;
        case 94:
        case 95:
        case 96:
        case 97:
        case 106:
        case 107:
        case 108:
        case 109:
          v16 = 10;
          break;
        default:
          v16 = 2;
          break;
      }
      goto LABEL_41;
    }
    goto LABEL_6;
  }
  v34 = sub_1F58D20((__int64)&v99);
  v16 = 0;
  if ( !v34 )
  {
LABEL_6:
    v18 = v100;
    goto LABEL_7;
  }
  v16 = sub_1F596B0((__int64)&v99);
LABEL_7:
  LOBYTE(v114) = v16;
  v115 = (__int64)v18;
  if ( !v16 )
  {
    v19 = sub_1F58D40((__int64)&v114);
    goto LABEL_9;
  }
LABEL_41:
  v19 = sub_2127930(v16);
LABEL_9:
  v20 = *(_QWORD *)(a2 + 72);
  v103 = v20;
  if ( v20 )
    sub_1623A60((__int64)&v103, v20, 2);
  v104 = *(_DWORD *)(a2 + 64);
  v21 = 32;
  if ( v19 )
  {
    _BitScanReverse(&v22, v19);
    v21 = v22 ^ 0x1F;
  }
  v106 = v17;
  v23 = v17 + v21 - 31;
  if ( v17 > 0x40 )
  {
    v89 = v23;
    sub_16A4EF0((__int64)&v105, 0, 0);
    v17 = v106;
    v23 = v89;
  }
  else
  {
    v105 = 0;
  }
  v24 = v17 - v23;
  if ( v17 - v23 != v17 )
  {
    if ( v24 > 0x3F || v17 > 0x40 )
      sub_16A5260(&v105, v24, v17);
    else
      v105 |= 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v23) << v24;
  }
  v25 = *(_QWORD *)(a2 + 32);
  v114 = 0;
  v115 = 1;
  v26 = *(_QWORD *)(a1 + 8);
  v116 = 0;
  v117 = 1;
  sub_1D1F820(v26, *(_QWORD *)(v25 + 40), *(_QWORD *)(v25 + 48), &v114, 0);
  LODWORD(v109) = v115;
  if ( (unsigned int)v115 <= 0x40 )
  {
    v27 = v114;
LABEL_21:
    v28 = (unsigned __int64)v116 | v27;
    LODWORD(v109) = 0;
    v108 = (_QWORD *)v28;
LABEL_22:
    v29 = v105 & v28;
LABEL_23:
    v30 = v29 == 0;
    goto LABEL_24;
  }
  sub_16A4FD0((__int64)&v108, (const void **)&v114);
  if ( (unsigned int)v109 <= 0x40 )
  {
    v27 = (__int64)v108;
    goto LABEL_21;
  }
  sub_16A89F0((__int64 *)&v108, (__int64 *)&v116);
  v58 = v109;
  v28 = (__int64)v108;
  LODWORD(v109) = 0;
  v111 = v58;
  v110 = v108;
  if ( v58 <= 0x40 )
    goto LABEL_22;
  sub_16A8890((__int64 *)&v110, (__int64 *)&v105);
  v59 = v111;
  v29 = (unsigned __int64)v110;
  v111 = 0;
  v113 = v59;
  v112 = (__int64)v110;
  if ( v59 <= 0x40 )
    goto LABEL_23;
  v94 = v110;
  if ( v59 - (unsigned int)sub_16A57B0((__int64)&v112) > 0x40 || *v94 )
  {
    if ( !v94 )
    {
      if ( (unsigned int)v109 > 0x40 && v108 )
        j_j___libc_free_0_0(v108);
      goto LABEL_46;
    }
    v30 = 0;
  }
  else
  {
    v30 = 1;
  }
  j_j___libc_free_0_0(v94);
  if ( v111 > 0x40 && v110 )
  {
    j_j___libc_free_0_0(v110);
    if ( (unsigned int)v109 <= 0x40 )
      goto LABEL_25;
    goto LABEL_44;
  }
LABEL_24:
  if ( (unsigned int)v109 <= 0x40 )
    goto LABEL_25;
LABEL_44:
  if ( v108 )
  {
    j_j___libc_free_0_0(v108);
    if ( v30 )
      goto LABEL_26;
    goto LABEL_46;
  }
LABEL_25:
  if ( v30 )
  {
LABEL_26:
    v31 = v117;
    v32 = 0;
    goto LABEL_27;
  }
LABEL_46:
  v35 = *(unsigned __int64 **)(a2 + 32);
  v107.m128i_i32[2] = 0;
  v108 = 0;
  LODWORD(v109) = 0;
  v36 = v35[1];
  v107.m128i_i64[0] = 0;
  sub_20174B0(a1, *v35, v36, &v107, &v108);
  v31 = v117;
  if ( (unsigned int)v117 <= 0x40 )
  {
    if ( (v105 & (unsigned __int64)v116) == 0 )
      goto LABEL_48;
LABEL_78:
    v62 = v106;
    v63 = *(__int64 **)(a1 + 8);
    v111 = v106;
    if ( v106 > 0x40 )
    {
      sub_16A4FD0((__int64)&v110, (const void **)&v105);
      v62 = v111;
      if ( v111 > 0x40 )
      {
        sub_16A8F40((__int64 *)&v110);
        v62 = v111;
        v65 = (unsigned __int64)v110;
LABEL_81:
        v112 = v65;
        v113 = v62;
        v111 = 0;
        *(_QWORD *)&v66 = sub_1D38970(
                            (__int64)v63,
                            (__int64)&v112,
                            (__int64)&v103,
                            v101,
                            v102,
                            0,
                            a5,
                            *(double *)v10.m128i_i64,
                            a7,
                            0);
        *(_QWORD *)&v97 = sub_1D332F0(
                            v63,
                            118,
                            (__int64)&v103,
                            v101,
                            v102,
                            0,
                            *(double *)a5.m128i_i64,
                            *(double *)v10.m128i_i64,
                            a7,
                            v10.m128i_i64[0],
                            v10.m128i_u64[1],
                            v66);
        *((_QWORD *)&v97 + 1) = v67 | v10.m128i_i64[1] & 0xFFFFFFFF00000000LL;
        if ( v113 > 0x40 && v112 )
          j_j___libc_free_0_0(v112);
        if ( v111 > 0x40 && v110 )
          j_j___libc_free_0_0(v110);
        v68 = *(_WORD *)(a2 + 24);
        if ( v68 == 123 )
        {
          v80 = *(__int64 **)(a1 + 8);
          *(_QWORD *)&v81 = sub_1D38BB0(
                              (__int64)v80,
                              v19 - 1,
                              (__int64)&v103,
                              v101,
                              v102,
                              0,
                              a5,
                              *(double *)v10.m128i_i64,
                              a7,
                              0);
          v82 = sub_1D332F0(
                  v80,
                  123,
                  (__int64)&v103,
                  v99,
                  v100,
                  0,
                  *(double *)a5.m128i_i64,
                  *(double *)v10.m128i_i64,
                  a7,
                  (__int64)v108,
                  v109,
                  v81);
          v83 = v99;
          v84 = v100;
          v32 = 1;
          a4->m128i_i64[0] = (__int64)v82;
          a4->m128i_i32[2] = v85;
          v86 = sub_1D332F0(
                  *(__int64 **)(a1 + 8),
                  123,
                  (__int64)&v103,
                  v83,
                  v84,
                  0,
                  *(double *)a5.m128i_i64,
                  *(double *)v10.m128i_i64,
                  a7,
                  (__int64)v108,
                  v109,
                  v97);
          v31 = v117;
          *(_QWORD *)a3 = v86;
          *(_DWORD *)(a3 + 8) = v87;
        }
        else
        {
          v69 = *(_QWORD *)(a1 + 8);
          if ( v68 == 124 )
          {
            v75 = sub_1D38BB0(v69, 0, (__int64)&v103, v99, v100, 0, a5, *(double *)v10.m128i_i64, a7, 0);
            v76 = v99;
            v32 = 1;
            v77 = v100;
            a4->m128i_i64[0] = v75;
            a4->m128i_i32[2] = v78;
            *(_QWORD *)a3 = sub_1D332F0(
                              *(__int64 **)(a1 + 8),
                              124,
                              (__int64)&v103,
                              v76,
                              v77,
                              0,
                              *(double *)a5.m128i_i64,
                              *(double *)v10.m128i_i64,
                              a7,
                              (__int64)v108,
                              v109,
                              v97);
            *(_DWORD *)(a3 + 8) = v79;
          }
          else
          {
            v70 = sub_1D38BB0(v69, 0, (__int64)&v103, v99, v100, 0, a5, *(double *)v10.m128i_i64, a7, 0);
            v71 = v99;
            v32 = 1;
            v72 = v100;
            *(_QWORD *)a3 = v70;
            *(_DWORD *)(a3 + 8) = v73;
            a4->m128i_i64[0] = (__int64)sub_1D332F0(
                                          *(__int64 **)(a1 + 8),
                                          122,
                                          (__int64)&v103,
                                          v71,
                                          v72,
                                          0,
                                          *(double *)a5.m128i_i64,
                                          *(double *)v10.m128i_i64,
                                          a7,
                                          v107.m128i_i64[0],
                                          v107.m128i_u64[1],
                                          v97);
            a4->m128i_i32[2] = v74;
          }
          v31 = v117;
        }
        goto LABEL_27;
      }
      v64 = (__int64)v110;
    }
    else
    {
      v64 = v105;
    }
    v65 = ~v64 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v62);
    v110 = (_QWORD *)v65;
    goto LABEL_81;
  }
  if ( (unsigned __int8)sub_16A59B0((__int64 *)&v116, (__int64 *)&v105) )
    goto LABEL_78;
LABEL_48:
  if ( v106 <= 0x40 )
  {
    if ( (v105 & ~v114) != 0 )
    {
      v32 = 0;
      goto LABEL_27;
    }
  }
  else
  {
    v32 = sub_16A5A00((__int64 *)&v105, (__int64 *)&v114);
    if ( !(_BYTE)v32 )
      goto LABEL_27;
  }
  v37 = *(__int64 **)(a1 + 8);
  *(_QWORD *)&v38 = sub_1D38BB0(
                      (__int64)v37,
                      v19 - 1,
                      (__int64)&v103,
                      v101,
                      v102,
                      0,
                      a5,
                      *(double *)v10.m128i_i64,
                      a7,
                      0);
  v40 = sub_1D332F0(
          v37,
          120,
          (__int64)&v103,
          v101,
          v102,
          0,
          *(double *)a5.m128i_i64,
          *(double *)v10.m128i_i64,
          a7,
          v10.m128i_i64[0],
          v10.m128i_u64[1],
          v38);
  if ( *(_WORD *)(a2 + 24) == 122 )
  {
    v41 = 124;
    v42 = 122;
  }
  else
  {
    a5 = _mm_loadu_si128(&v107);
    v41 = 122;
    v42 = 124;
    v107.m128i_i64[0] = (__int64)v108;
    v107.m128i_i32[2] = v109;
    v108 = (_QWORD *)a5.m128i_i64[0];
    LODWORD(v109) = a5.m128i_i32[2];
  }
  *(_QWORD *)&v88 = v40;
  *((_QWORD *)&v88 + 1) = v39;
  v90 = *(__int64 **)(a1 + 8);
  *(_QWORD *)&v43 = sub_1D38BB0((__int64)v90, 1, (__int64)&v103, v101, v102, 0, a5, *(double *)v10.m128i_i64, a7, 0);
  v44 = sub_1D332F0(
          v90,
          v41,
          (__int64)&v103,
          v99,
          v100,
          0,
          *(double *)a5.m128i_i64,
          *(double *)v10.m128i_i64,
          a7,
          v107.m128i_i64[0],
          v107.m128i_u64[1],
          v43);
  *(_QWORD *)&v46 = sub_1D332F0(
                      *(__int64 **)(a1 + 8),
                      v41,
                      (__int64)&v103,
                      v99,
                      v100,
                      0,
                      *(double *)a5.m128i_i64,
                      *(double *)v10.m128i_i64,
                      a7,
                      (__int64)v44,
                      v45,
                      v88);
  v91 = v46;
  v98 = sub_1D332F0(
          *(__int64 **)(a1 + 8),
          *(unsigned __int16 *)(a2 + 24),
          (__int64)&v103,
          v99,
          v100,
          0,
          *(double *)a5.m128i_i64,
          *(double *)v10.m128i_i64,
          a7,
          v107.m128i_i64[0],
          v107.m128i_u64[1],
          *(_OWORD *)&v10);
  v47 = v99;
  *(_QWORD *)a3 = v98;
  v48 = v100;
  *(_DWORD *)(a3 + 8) = v49;
  v50 = *(__int64 **)(a1 + 8);
  v51 = sub_1D332F0(
          v50,
          v42,
          (__int64)&v103,
          v47,
          v48,
          0,
          *(double *)a5.m128i_i64,
          *(double *)v10.m128i_i64,
          a7,
          (__int64)v108,
          v109,
          *(_OWORD *)&v10);
  a4->m128i_i64[0] = (__int64)sub_1D332F0(
                                v50,
                                119,
                                (__int64)&v103,
                                v99,
                                v100,
                                0,
                                *(double *)a5.m128i_i64,
                                *(double *)v10.m128i_i64,
                                a7,
                                (__int64)v51,
                                v52,
                                v91);
  a4->m128i_i32[2] = v53;
  if ( *(_WORD *)(a2 + 24) != 122 )
  {
    v54 = _mm_loadu_si128(a4);
    a4->m128i_i64[0] = *(_QWORD *)a3;
    a4->m128i_i32[2] = *(_DWORD *)(a3 + 8);
    *(_QWORD *)a3 = v54.m128i_i64[0];
    *(_DWORD *)(a3 + 8) = v54.m128i_i32[2];
  }
  v31 = v117;
  v32 = 1;
LABEL_27:
  if ( v31 > 0x40 && v116 )
    j_j___libc_free_0_0(v116);
  if ( (unsigned int)v115 > 0x40 && v114 )
    j_j___libc_free_0_0(v114);
  if ( v106 > 0x40 && v105 )
    j_j___libc_free_0_0(v105);
  if ( v103 )
    sub_161E7C0((__int64)&v103, v103);
  return v32;
}
