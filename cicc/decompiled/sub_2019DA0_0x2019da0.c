// Function: sub_2019DA0
// Address: 0x2019da0
//
__int64 __fastcall sub_2019DA0(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rbx
  __int64 v9; // rax
  __int64 v10; // rsi
  unsigned __int8 v11; // r13
  __int64 v12; // rax
  __m128 v13; // xmm0
  __int128 v14; // xmm1
  __int128 v15; // xmm2
  _BYTE *v16; // rax
  __int64 v17; // r12
  char v19; // di
  unsigned int v20; // eax
  char v21; // di
  char v22; // r8
  int v23; // esi
  __int8 v24; // al
  char v25; // r13
  const void **v26; // r8
  __int64 *v27; // r14
  __int64 v28; // rdx
  unsigned int v29; // eax
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // r9
  __int64 v33; // r8
  unsigned __int64 v34; // rcx
  const void **v35; // rsi
  __int64 v36; // rdx
  char v37; // al
  __int64 v38; // rdx
  unsigned int v39; // r11d
  __int64 v40; // r9
  unsigned int v41; // edx
  __int64 *v42; // r8
  __int64 *v43; // r11
  unsigned int v44; // r9d
  int v45; // eax
  __int64 v46; // rdx
  _BYTE *v47; // rcx
  _BYTE *v48; // rsi
  __int64 *v49; // rax
  unsigned int v50; // edx
  unsigned int v51; // ecx
  __int64 *v52; // r10
  unsigned int v53; // edx
  unsigned int v54; // edx
  __int64 *v55; // r14
  unsigned int v56; // eax
  __int128 v57; // rax
  __int128 v58; // rax
  unsigned int v59; // edx
  unsigned int v60; // edx
  __int128 v61; // rax
  bool v62; // al
  const void **v63; // rdx
  __int64 *v64; // rax
  unsigned int v65; // edx
  unsigned int v66; // eax
  __int128 v67; // [rsp-20h] [rbp-280h]
  __int128 v68; // [rsp-10h] [rbp-270h]
  __int128 v69; // [rsp-10h] [rbp-270h]
  const void **v70; // [rsp+8h] [rbp-258h]
  unsigned int v71; // [rsp+8h] [rbp-258h]
  __int64 v72; // [rsp+10h] [rbp-250h]
  __int64 *v73; // [rsp+10h] [rbp-250h]
  __int64 v74; // [rsp+18h] [rbp-248h]
  __int64 v75; // [rsp+20h] [rbp-240h]
  unsigned int v76; // [rsp+20h] [rbp-240h]
  unsigned int v77; // [rsp+20h] [rbp-240h]
  __int64 *v78; // [rsp+20h] [rbp-240h]
  __int64 *v79; // [rsp+20h] [rbp-240h]
  __int64 v80; // [rsp+28h] [rbp-238h]
  unsigned __int64 v81; // [rsp+30h] [rbp-230h]
  __int64 *v82; // [rsp+38h] [rbp-228h]
  __int64 *v83; // [rsp+38h] [rbp-228h]
  __int64 v84; // [rsp+38h] [rbp-228h]
  __int64 *v85; // [rsp+40h] [rbp-220h]
  __int128 v86; // [rsp+40h] [rbp-220h]
  __int128 v87; // [rsp+40h] [rbp-220h]
  __int64 v88; // [rsp+50h] [rbp-210h]
  __int64 *v89; // [rsp+50h] [rbp-210h]
  unsigned __int64 v90; // [rsp+58h] [rbp-208h]
  __int64 v91; // [rsp+60h] [rbp-200h]
  __int64 *v92; // [rsp+60h] [rbp-200h]
  unsigned __int64 v93; // [rsp+68h] [rbp-1F8h]
  unsigned __int64 v94; // [rsp+68h] [rbp-1F8h]
  __int64 *v95; // [rsp+70h] [rbp-1F0h]
  __int128 v96; // [rsp+70h] [rbp-1F0h]
  unsigned __int64 v97; // [rsp+78h] [rbp-1E8h]
  __int64 v98; // [rsp+E0h] [rbp-180h] BYREF
  int v99; // [rsp+E8h] [rbp-178h]
  __m128i v100; // [rsp+F0h] [rbp-170h] BYREF
  unsigned __int64 v101; // [rsp+100h] [rbp-160h] BYREF
  const void **v102; // [rsp+108h] [rbp-158h]
  __m128i v103; // [rsp+110h] [rbp-150h] BYREF
  _BYTE *v104; // [rsp+120h] [rbp-140h] BYREF
  __int64 v105; // [rsp+128h] [rbp-138h]
  _BYTE v106[304]; // [rsp+130h] [rbp-130h] BYREF

  v8 = 16LL * a3;
  v9 = *(_QWORD *)(a2 + 40);
  v10 = *(_QWORD *)(a2 + 72);
  v11 = *(_BYTE *)(v9 + v8);
  v98 = v10;
  if ( v10 )
    sub_1623A60((__int64)&v98, v10, 2);
  v99 = *(_DWORD *)(a2 + 64);
  v12 = *(_QWORD *)(a2 + 32);
  v13 = (__m128)_mm_loadu_si128((const __m128i *)v12);
  v14 = (__int128)_mm_loadu_si128((const __m128i *)(v12 + 40));
  v15 = (__int128)_mm_loadu_si128((const __m128i *)(v12 + 80));
  if ( !v11
    || (v16 = (_BYTE *)(*(_QWORD *)(a1 + 8) + 259LL * v11), v16[2540] == 2)
    || v16[2542] == 2
    || v16[2541] == 2
    || v16[2526] == 2 )
  {
    v17 = (__int64)sub_1D40890(*(__int64 **)a1, a2, 0, a4, a5, a6, (__m128i)v13, *(double *)&v14, (__m128i)v15);
    goto LABEL_9;
  }
  switch ( v11 )
  {
    case 0xEu:
    case 0xFu:
    case 0x10u:
    case 0x11u:
    case 0x12u:
    case 0x13u:
    case 0x14u:
    case 0x15u:
    case 0x16u:
    case 0x17u:
    case 0x38u:
    case 0x39u:
    case 0x3Au:
    case 0x3Bu:
    case 0x3Cu:
    case 0x3Du:
      v19 = 2;
      goto LABEL_14;
    case 0x18u:
    case 0x19u:
    case 0x1Au:
    case 0x1Bu:
    case 0x1Cu:
    case 0x1Du:
    case 0x1Eu:
    case 0x1Fu:
    case 0x20u:
    case 0x3Eu:
    case 0x3Fu:
    case 0x40u:
    case 0x41u:
    case 0x42u:
    case 0x43u:
      v19 = 3;
      goto LABEL_14;
    case 0x21u:
    case 0x22u:
    case 0x23u:
    case 0x24u:
    case 0x25u:
    case 0x26u:
    case 0x27u:
    case 0x28u:
    case 0x44u:
    case 0x45u:
    case 0x46u:
    case 0x47u:
    case 0x48u:
    case 0x49u:
      v19 = 4;
      goto LABEL_14;
    case 0x29u:
    case 0x2Au:
    case 0x2Bu:
    case 0x2Cu:
    case 0x2Du:
    case 0x2Eu:
    case 0x2Fu:
    case 0x30u:
    case 0x4Au:
    case 0x4Bu:
    case 0x4Cu:
    case 0x4Du:
    case 0x4Eu:
    case 0x4Fu:
      v19 = 5;
      goto LABEL_14;
    case 0x31u:
    case 0x32u:
    case 0x33u:
    case 0x34u:
    case 0x35u:
    case 0x36u:
    case 0x50u:
    case 0x51u:
    case 0x52u:
    case 0x53u:
    case 0x54u:
    case 0x55u:
      v19 = 6;
      goto LABEL_14;
    case 0x37u:
      v20 = sub_2018C90(7);
      if ( v20 != 32 )
        goto LABEL_15;
      v22 = 5;
      v23 = 1;
      goto LABEL_21;
    case 0x56u:
    case 0x57u:
    case 0x58u:
    case 0x62u:
    case 0x63u:
    case 0x64u:
      v19 = 8;
      goto LABEL_14;
    case 0x59u:
    case 0x5Au:
    case 0x5Bu:
    case 0x5Cu:
    case 0x5Du:
    case 0x65u:
    case 0x66u:
    case 0x67u:
    case 0x68u:
    case 0x69u:
      v19 = 9;
      goto LABEL_14;
    case 0x5Eu:
    case 0x5Fu:
    case 0x60u:
    case 0x61u:
    case 0x6Au:
    case 0x6Bu:
    case 0x6Cu:
    case 0x6Du:
      v19 = 10;
LABEL_14:
      v20 = sub_2018C90(v19);
      if ( v20 == 32 )
      {
        v21 = 5;
      }
      else
      {
LABEL_15:
        if ( v20 > 0x20 )
        {
          v21 = 6;
          if ( v20 != 64 )
          {
            v21 = 7;
            if ( v20 != 128 )
              v21 = 0;
          }
        }
        else
        {
          v21 = 3;
          if ( v20 != 8 )
          {
            v21 = 4;
            if ( v20 != 16 )
              v21 = 2 * (v20 == 1);
          }
        }
      }
      v22 = v21;
      v23 = word_4301260[(unsigned __int8)(v11 - 14)];
      if ( (unsigned __int8)(v11 - 98) <= 0xBu || (unsigned __int8)(v11 - 56) <= 0x1Du )
        v24 = sub_1D154A0(v21, v23);
      else
LABEL_21:
        v24 = sub_1D15020(v22, v23);
      v100.m128i_i8[0] = v24;
      v25 = v24;
      v100.m128i_i64[1] = 0;
      if ( v24 )
      {
        if ( (unsigned __int8)(v24 - 14) <= 0x5Fu )
        {
          switch ( v24 )
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
              v25 = 3;
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
              v25 = 4;
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
              v25 = 5;
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
              v25 = 6;
              break;
            case 55:
              v25 = 7;
              break;
            case 86:
            case 87:
            case 88:
            case 98:
            case 99:
            case 100:
              v25 = 8;
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
              v25 = 9;
              break;
            case 94:
            case 95:
            case 96:
            case 97:
            case 106:
            case 107:
            case 108:
            case 109:
              v25 = 10;
              break;
            default:
              v25 = 2;
              break;
          }
        }
        goto LABEL_24;
      }
      if ( !sub_1F58D20((__int64)&v100) )
      {
LABEL_24:
        v26 = 0;
        goto LABEL_25;
      }
      v25 = sub_1F596B0((__int64)&v100);
      v26 = v63;
LABEL_25:
      v27 = *(__int64 **)a1;
      LOBYTE(v101) = v25;
      v102 = v26;
      v75 = sub_1D38BB0(
              (__int64)v27,
              0,
              (__int64)&v98,
              (unsigned int)v101,
              v26,
              0,
              (__m128i)v13,
              *(double *)&v14,
              (__m128i)v15,
              0);
      v80 = v28;
      v85 = *(__int64 **)a1;
      if ( (_BYTE)v101 )
        v29 = sub_2018C90(v101);
      else
        v29 = sub_1F58D40((__int64)&v101);
      v103.m128i_i32[2] = v29;
      if ( v29 > 0x40 )
        sub_16A4EF0((__int64)&v103, -1, 1);
      else
        v103.m128i_i64[0] = 0xFFFFFFFFFFFFFFFFLL >> -(char)v29;
      v30 = sub_1D38970(
              (__int64)v85,
              (__int64)&v103,
              (__int64)&v98,
              v101,
              v102,
              0,
              (__m128i)v13,
              *(double *)&v14,
              (__m128i)v15,
              0);
      v32 = v31;
      v33 = v30;
      v34 = v101;
      v35 = v102;
      v36 = *(_QWORD *)(v13.m128_u64[0] + 40) + 16LL * v13.m128_u32[2];
      v37 = *(_BYTE *)v36;
      v38 = *(_QWORD *)(v36 + 8);
      LOBYTE(v104) = v37;
      v105 = v38;
      if ( v37 )
      {
        if ( (unsigned __int8)(v37 - 14) > 0x5Fu )
          v39 = 134;
        else
          v39 = 135;
      }
      else
      {
        v70 = v102;
        v81 = v101;
        v72 = v33;
        v74 = v32;
        v62 = sub_1F58D20((__int64)&v104);
        v33 = v72;
        v32 = v74;
        v34 = v81;
        v35 = v70;
        v39 = 134 - (!v62 - 1);
      }
      *((_QWORD *)&v67 + 1) = v32;
      *(_QWORD *)&v67 = v33;
      v95 = sub_1D3A900(
              v27,
              v39,
              (__int64)&v98,
              v34,
              v35,
              0,
              v13,
              *(double *)&v14,
              (__m128i)v15,
              v13.m128_u64[0],
              (__int16 *)v13.m128_u64[1],
              v67,
              v75,
              v80);
      v97 = v41 | v13.m128_u64[1] & 0xFFFFFFFF00000000LL;
      if ( v103.m128i_i32[2] > 0x40u && v103.m128i_i64[0] )
        j_j___libc_free_0_0(v103.m128i_i64[0]);
      v42 = v95;
      v43 = *(__int64 **)a1;
      v103 = _mm_loadu_si128(&v100);
      if ( *((_WORD *)v95 + 12) == 48 )
      {
        v104 = 0;
        LODWORD(v105) = 0;
        v64 = sub_1D2B300(v43, 0x30u, (__int64)&v104, v103.m128i_u32[0], v103.m128i_i64[1], v40);
        v51 = v65;
        v52 = v64;
        if ( v104 )
        {
          v77 = v65;
          v83 = v64;
          sub_161E7C0((__int64)&v104, (__int64)v104);
          v52 = v83;
          v51 = v77;
        }
      }
      else
      {
        if ( v103.m128i_i8[0] )
        {
          v44 = word_4301260[(unsigned __int8)(v103.m128i_i8[0] - 14)];
        }
        else
        {
          v78 = v43;
          v66 = sub_1F58D30((__int64)&v103);
          v42 = v95;
          v43 = v78;
          v44 = v66;
        }
        v45 = v97;
        v46 = v44;
        v105 = 0x1000000000LL;
        v47 = v106;
        v104 = v106;
        if ( v44 > 0x10 )
        {
          v71 = v44;
          v73 = v42;
          v79 = v43;
          v84 = v44;
          sub_16CD150((__int64)&v104, v106, v44, 16, (int)v42, v44);
          v47 = v104;
          v44 = v71;
          v45 = v97;
          v42 = v73;
          v43 = v79;
          v46 = v84;
        }
        LODWORD(v105) = v44;
        v48 = &v47[16 * v46];
        if ( v48 != v47 )
        {
          do
          {
            if ( v47 )
            {
              *(_QWORD *)v47 = v42;
              *((_DWORD *)v47 + 2) = v45;
            }
            v47 += 16;
          }
          while ( v48 != v47 );
          v47 = v104;
          v46 = (unsigned int)v105;
        }
        *((_QWORD *)&v68 + 1) = v46;
        *(_QWORD *)&v68 = v47;
        v49 = sub_1D359D0(
                v43,
                104,
                (__int64)&v98,
                v103.m128i_u32[0],
                (const void **)v103.m128i_i64[1],
                0,
                *(double *)v13.m128_u64,
                *(double *)&v14,
                (__m128i)v15,
                v68);
        v51 = v50;
        v52 = v49;
        if ( v104 != v106 )
        {
          v76 = v50;
          v82 = v49;
          _libc_free((unsigned __int64)v104);
          v51 = v76;
          v52 = v82;
        }
      }
      *(_QWORD *)&v96 = v52;
      *((_QWORD *)&v96 + 1) = v51 | v97 & 0xFFFFFFFF00000000LL;
      v91 = sub_1D309E0(
              *(__int64 **)a1,
              158,
              (__int64)&v98,
              v100.m128i_u32[0],
              (const void **)v100.m128i_i64[1],
              0,
              *(double *)v13.m128_u64,
              *(double *)&v14,
              *(double *)&v15,
              v14);
      v93 = v53 | *((_QWORD *)&v14 + 1) & 0xFFFFFFFF00000000LL;
      v88 = sub_1D309E0(
              *(__int64 **)a1,
              158,
              (__int64)&v98,
              v100.m128i_u32[0],
              (const void **)v100.m128i_i64[1],
              0,
              *(double *)v13.m128_u64,
              *(double *)&v14,
              *(double *)&v15,
              v15);
      v90 = v54 | *((_QWORD *)&v15 + 1) & 0xFFFFFFFF00000000LL;
      v55 = *(__int64 **)a1;
      if ( (_BYTE)v101 )
        v56 = sub_2018C90(v101);
      else
        v56 = sub_1F58D40((__int64)&v101);
      LODWORD(v105) = v56;
      if ( v56 > 0x40 )
        sub_16A4EF0((__int64)&v104, -1, 1);
      else
        v104 = (_BYTE *)(0xFFFFFFFFFFFFFFFFLL >> -(char)v56);
      *(_QWORD *)&v57 = sub_1D38970(
                          (__int64)v55,
                          (__int64)&v104,
                          (__int64)&v98,
                          v100.m128i_u32[0],
                          (const void **)v100.m128i_i64[1],
                          0,
                          (__m128i)v13,
                          *(double *)&v14,
                          (__m128i)v15,
                          0);
      if ( (unsigned int)v105 > 0x40 && v104 )
      {
        v86 = v57;
        j_j___libc_free_0_0(v104);
        v57 = v86;
      }
      *(_QWORD *)&v58 = sub_1D332F0(
                          *(__int64 **)a1,
                          120,
                          (__int64)&v98,
                          v100.m128i_u32[0],
                          (const void **)v100.m128i_i64[1],
                          0,
                          *(double *)v13.m128_u64,
                          *(double *)&v14,
                          (__m128i)v15,
                          v96,
                          *((unsigned __int64 *)&v96 + 1),
                          v57);
      v87 = v58;
      v92 = sub_1D332F0(
              *(__int64 **)a1,
              118,
              (__int64)&v98,
              v100.m128i_u32[0],
              (const void **)v100.m128i_i64[1],
              0,
              *(double *)v13.m128_u64,
              *(double *)&v14,
              (__m128i)v15,
              v91,
              v93,
              v96);
      v94 = v59 | v93 & 0xFFFFFFFF00000000LL;
      v89 = sub_1D332F0(
              *(__int64 **)a1,
              118,
              (__int64)&v98,
              v100.m128i_u32[0],
              (const void **)v100.m128i_i64[1],
              0,
              *(double *)v13.m128_u64,
              *(double *)&v14,
              (__m128i)v15,
              v88,
              v90,
              v87);
      *((_QWORD *)&v69 + 1) = v60 | v90 & 0xFFFFFFFF00000000LL;
      *(_QWORD *)&v69 = v89;
      *(_QWORD *)&v61 = sub_1D332F0(
                          *(__int64 **)a1,
                          119,
                          (__int64)&v98,
                          v100.m128i_u32[0],
                          (const void **)v100.m128i_i64[1],
                          0,
                          *(double *)v13.m128_u64,
                          *(double *)&v14,
                          (__m128i)v15,
                          (__int64)v92,
                          v94,
                          v69);
      v17 = sub_1D309E0(
              *(__int64 **)a1,
              158,
              (__int64)&v98,
              *(unsigned __int8 *)(*(_QWORD *)(a2 + 40) + v8),
              *(const void ***)(*(_QWORD *)(a2 + 40) + v8 + 8),
              0,
              *(double *)v13.m128_u64,
              *(double *)&v14,
              *(double *)&v15,
              v61);
LABEL_9:
      if ( v98 )
        sub_161E7C0((__int64)&v98, v98);
      return v17;
  }
}
