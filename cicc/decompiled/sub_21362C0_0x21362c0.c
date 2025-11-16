// Function: sub_21362C0
// Address: 0x21362c0
//
__int64 *__fastcall sub_21362C0(__int64 *a1, _QWORD *a2, double a3, __m128i a4, __m128i a5)
{
  _QWORD *v5; // rcx
  __m128 v7; // xmm0
  __int64 v8; // rax
  unsigned __int8 v9; // r13
  __int64 v10; // rax
  __int64 v11; // rsi
  unsigned __int8 v12; // r12
  const void **v13; // r15
  int v14; // eax
  char v15; // al
  void *v16; // rdi
  int v17; // eax
  unsigned int v18; // ecx
  int v19; // eax
  __int64 *v20; // r14
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rax
  unsigned int v25; // edx
  __int64 v26; // rax
  __int64 *v27; // r13
  const void ***v28; // rax
  __int128 v29; // rax
  __int64 v30; // r14
  __int64 v31; // rsi
  __int64 v32; // rax
  unsigned __int64 v33; // rax
  const void **v34; // rdx
  __int64 *v35; // rax
  __int64 v36; // r13
  __int16 *v37; // rdx
  __int64 v38; // rax
  unsigned int v39; // eax
  unsigned __int8 v40; // r11
  unsigned int v41; // r14d
  __int64 *v42; // rax
  __int64 v43; // rax
  unsigned int v44; // r10d
  unsigned __int64 v45; // rdx
  __int64 v46; // rax
  __int64 v47; // rdx
  __int64 v48; // r13
  __int64 v49; // rdx
  __int64 v50; // r14
  _BYTE *v51; // rax
  __int64 v52; // rsi
  __int64 v53; // r11
  unsigned int v54; // eax
  __int64 *v55; // rax
  __int64 v56; // rdx
  __int64 v57; // r14
  const void ***v58; // rdx
  __int64 *v59; // rax
  _QWORD *v60; // r13
  __int64 v61; // rsi
  unsigned int v62; // edx
  __int64 v63; // r14
  int v64; // r9d
  __int128 v65; // rax
  __int128 v66; // [rsp-10h] [rbp-170h]
  __int128 v67; // [rsp+0h] [rbp-160h]
  unsigned int v68; // [rsp+10h] [rbp-150h]
  __int64 v69; // [rsp+18h] [rbp-148h]
  __int64 v70; // [rsp+28h] [rbp-138h]
  unsigned __int64 v71; // [rsp+30h] [rbp-130h]
  __int16 *v72; // [rsp+38h] [rbp-128h]
  __int64 v73; // [rsp+40h] [rbp-120h]
  __int64 v74; // [rsp+48h] [rbp-118h]
  __int64 v75; // [rsp+48h] [rbp-118h]
  unsigned int v76; // [rsp+48h] [rbp-118h]
  __int64 v77; // [rsp+50h] [rbp-110h]
  unsigned int v78; // [rsp+50h] [rbp-110h]
  __int64 (__fastcall *v79)(__int64, __int64, __int64, __int64, __int64); // [rsp+58h] [rbp-108h]
  __int64 v80; // [rsp+60h] [rbp-100h]
  __int64 v81; // [rsp+68h] [rbp-F8h]
  unsigned __int64 v82; // [rsp+68h] [rbp-F8h]
  _QWORD *v83; // [rsp+70h] [rbp-F0h]
  __int64 v84; // [rsp+70h] [rbp-F0h]
  __int64 v85; // [rsp+70h] [rbp-F0h]
  __int64 v86; // [rsp+70h] [rbp-F0h]
  __int64 v87; // [rsp+78h] [rbp-E8h]
  __int64 v88; // [rsp+80h] [rbp-E0h]
  __int128 v89; // [rsp+80h] [rbp-E0h]
  __int64 v90; // [rsp+80h] [rbp-E0h]
  unsigned __int64 v91; // [rsp+88h] [rbp-D8h]
  __m128 v92; // [rsp+A0h] [rbp-C0h] BYREF
  __int64 v93; // [rsp+B0h] [rbp-B0h] BYREF
  int v94; // [rsp+B8h] [rbp-A8h]
  __int64 v95; // [rsp+C0h] [rbp-A0h] BYREF
  unsigned int v96; // [rsp+C8h] [rbp-98h]
  __int64 v97; // [rsp+D0h] [rbp-90h] BYREF
  int v98; // [rsp+D8h] [rbp-88h]
  unsigned __int64 v99; // [rsp+E0h] [rbp-80h] BYREF
  __int16 *v100; // [rsp+E8h] [rbp-78h]
  _QWORD v101[4]; // [rsp+F0h] [rbp-70h] BYREF
  __int128 v102; // [rsp+110h] [rbp-50h] BYREF
  __int64 v103; // [rsp+120h] [rbp-40h]

  v5 = a2;
  v7 = (__m128)_mm_loadu_si128((const __m128i *)a2[4]);
  v92 = v7;
  v8 = *(_QWORD *)(v7.m128_u64[0] + 40) + 16LL * v7.m128_u32[2];
  v9 = *(_BYTE *)v8;
  v88 = *(_QWORD *)(v8 + 8);
  v10 = a2[5];
  v11 = a2[9];
  v12 = *(_BYTE *)v10;
  v13 = *(const void ***)(v10 + 8);
  v93 = v11;
  if ( v11 )
  {
    v83 = v5;
    sub_1623A60((__int64)&v93, v11, 2);
    v5 = v83;
  }
  v14 = *((_DWORD *)v5 + 16);
  LOBYTE(v102) = v12;
  *((_QWORD *)&v102 + 1) = v13;
  v94 = v14;
  if ( v12 )
  {
    if ( (unsigned __int8)(v12 - 14) > 0x5Fu )
    {
      v15 = v12;
LABEL_6:
      switch ( v15 )
      {
        case 8:
          goto LABEL_22;
        case 9:
          goto LABEL_21;
        case 10:
          goto LABEL_20;
        case 11:
          v16 = sub_16982A0();
          goto LABEL_11;
        case 12:
          v16 = sub_1698290();
          goto LABEL_11;
        case 13:
          v16 = sub_16982C0();
          goto LABEL_11;
        default:
          goto LABEL_48;
      }
    }
    switch ( v12 )
    {
      case 'V':
      case 'W':
      case 'X':
      case 'b':
      case 'c':
      case 'd':
LABEL_22:
        v16 = sub_1698260();
        goto LABEL_11;
      case 'Y':
      case 'Z':
      case '[':
      case '\\':
      case ']':
      case 'e':
      case 'f':
      case 'g':
      case 'h':
      case 'i':
LABEL_21:
        v16 = sub_1698270();
        goto LABEL_11;
      case '^':
      case '_':
      case '`':
      case 'a':
      case 'j':
      case 'k':
      case 'l':
      case 'm':
LABEL_20:
        v16 = sub_1698280();
LABEL_11:
        sub_16982D0((__int64)v16);
        if ( v9 && (v17 = sub_2127930(v9), v18 >= v17 - 1) && *(_BYTE *)(*a1 + 259LL * v9 + 2568) == 4 )
        {
          v22 = sub_1D309E0(
                  (__int64 *)a1[1],
                  146,
                  (__int64)&v93,
                  v12,
                  v13,
                  0,
                  *(double *)v7.m128_u64,
                  *(double *)a4.m128i_i64,
                  *(double *)a5.m128i_i64,
                  *(_OWORD *)&v92);
          v81 = v23;
          v24 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)*a1 + 1312LL))(
                  *a1,
                  v22,
                  v23,
                  a1[1]);
          v96 = 32;
          v80 = v24;
          v82 = v25 | v81 & 0xFFFFFFFF00000000LL;
          if ( v9 == 5 )
          {
            v95 = 1333788672;
          }
          else
          {
            v26 = 2139095040;
            if ( v9 == 6 )
              v26 = 1602224128;
            v95 = v26;
          }
          v97 = 0;
          v98 = 0;
          v99 = 0;
          LODWORD(v100) = 0;
          sub_20174B0((__int64)a1, v92.m128_u64[0], v92.m128_i64[1], &v97, &v99);
          v27 = (__int64 *)a1[1];
          v28 = (const void ***)(*(_QWORD *)(v99 + 40) + 16LL * (unsigned int)v100);
          *(_QWORD *)&v29 = sub_1D38BB0(
                              (__int64)v27,
                              0,
                              (__int64)&v93,
                              *(unsigned __int8 *)v28,
                              v28[1],
                              0,
                              (__m128i)v7,
                              *(double *)a4.m128i_i64,
                              a5,
                              0);
          v30 = *a1;
          v31 = a1[1];
          v89 = v29;
          *(_QWORD *)&v29 = *(_QWORD *)(v99 + 40) + 16LL * (unsigned int)v100;
          v84 = *(_QWORD *)(v31 + 48);
          v74 = *(_QWORD *)(v29 + 8);
          v77 = *(unsigned __int8 *)v29;
          v79 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)*a1 + 264LL);
          v32 = sub_1E0A0C0(*(_QWORD *)(v31 + 32));
          v33 = v79(v30, v32, v84, v77, v74);
          v35 = sub_1F81070(v27, (__int64)&v93, v33, v34, v99, v100, v7, *(double *)a4.m128i_i64, a5, v89, 0x14u);
          v36 = a1[1];
          v72 = v37;
          v71 = (unsigned __int64)v35;
          v38 = sub_1E0A0C0(*(_QWORD *)(v36 + 32));
          v39 = 8 * sub_15A9520(v38, 0);
          if ( v39 == 32 )
          {
            v40 = 5;
          }
          else if ( v39 > 0x20 )
          {
            v40 = 6;
            if ( v39 != 64 )
            {
              v40 = 0;
              if ( v39 == 128 )
                v40 = 7;
            }
          }
          else
          {
            v40 = 3;
            if ( v39 != 8 )
            {
              v40 = 4;
              if ( v39 != 16 )
                v40 = 0;
            }
          }
          v41 = v40;
          sub_16A5C50((__int64)&v102, (const void **)&v95, 0x40u);
          v42 = (__int64 *)sub_159C0E0(*(__int64 **)(a1[1] + 48), (__int64)&v102);
          v43 = sub_1D2A150(v36, v42, v41, 0, 0, 0, 0, 0);
          v44 = v12;
          v90 = v43;
          v91 = v45;
          v75 = v43;
          v78 = v45;
          if ( DWORD2(v102) > 0x40 && (_QWORD)v102 )
          {
            j_j___libc_free_0_0(v102);
            v44 = v12;
          }
          v68 = v44;
          v46 = sub_1D38E70(a1[1], 0, (__int64)&v93, 0, (__m128i)v7, *(double *)a4.m128i_i64, a5);
          v87 = v47;
          v48 = (unsigned int)v47;
          v69 = v46;
          v85 = v46;
          v73 = sub_1D38E70(a1[1], 4, (__int64)&v93, 0, (__m128i)v7, *(double *)a4.m128i_i64, a5);
          v50 = (unsigned int)v49;
          v70 = v49;
          v51 = (_BYTE *)sub_1E0A0C0(*(_QWORD *)(a1[1] + 32));
          v52 = v73;
          v53 = v69;
          if ( *v51 )
          {
            v54 = v48;
            v52 = v85;
            v48 = (unsigned int)v50;
            v53 = v73;
            v50 = v54;
          }
          *((_QWORD *)&v66 + 1) = v48 | v87 & 0xFFFFFFFF00000000LL;
          *(_QWORD *)&v66 = v53;
          v55 = sub_1F810E0(
                  (__int64 *)a1[1],
                  (__int64)&v93,
                  *(unsigned __int8 *)(*(_QWORD *)(v53 + 40) + 16 * v48),
                  *(const void ***)(*(_QWORD *)(v53 + 40) + 16 * v48 + 8),
                  v71,
                  v72,
                  v7,
                  *(double *)a4.m128i_i64,
                  a5,
                  v66,
                  v52,
                  v50 | v70 & 0xFFFFFFFF00000000LL);
          v57 = v56;
          v58 = (const void ***)(*(_QWORD *)(v75 + 40) + 16LL * v78);
          *((_QWORD *)&v67 + 1) = v57;
          *(_QWORD *)&v67 = v55;
          v76 = *(_DWORD *)(v75 + 100);
          v59 = sub_1D332F0(
                  (__int64 *)a1[1],
                  52,
                  (__int64)&v93,
                  *(unsigned __int8 *)v58,
                  v58[1],
                  0,
                  *(double *)v7.m128_u64,
                  *(double *)a4.m128i_i64,
                  a5,
                  v90,
                  v91,
                  v67);
          v60 = (_QWORD *)a1[1];
          v86 = (__int64)v59;
          v61 = v60[4];
          memset(v101, 0, 24);
          v63 = v62;
          sub_1E34190((__int64)&v102, v61);
          v64 = 4;
          if ( v76 <= 4 )
            v64 = v76;
          *(_QWORD *)&v65 = sub_1D2B810(
                              v60,
                              1u,
                              (__int64)&v93,
                              v68,
                              (__int64)v13,
                              v64,
                              (unsigned __int64)(a1[1] + 88),
                              v86,
                              v63 | v91 & 0xFFFFFFFF00000000LL,
                              v102,
                              v103,
                              9,
                              0,
                              0,
                              (__int64)v101);
          v20 = sub_1D332F0(
                  (__int64 *)a1[1],
                  76,
                  (__int64)&v93,
                  v68,
                  v13,
                  0,
                  *(double *)v7.m128_u64,
                  *(double *)a4.m128i_i64,
                  a5,
                  v80,
                  v82,
                  v65);
          if ( v96 > 0x40 && v95 )
            j_j___libc_free_0_0(v95);
        }
        else
        {
          v19 = sub_1F402E0(v9, v88, v12);
          sub_20BE530(
            (__int64)&v102,
            (__m128i *)*a1,
            a1[1],
            v19,
            v12,
            (__int64)v13,
            (__m128i)v7,
            a4,
            a5,
            (__int64)&v92,
            1u,
            1u,
            (__int64)&v93,
            0,
            1);
          v20 = (__int64 *)v102;
        }
        if ( v93 )
          sub_161E7C0((__int64)&v93, v93);
        return v20;
      default:
        break;
    }
  }
  else if ( sub_1F58D20((__int64)&v102) )
  {
    v15 = sub_1F596B0((__int64)&v102);
    goto LABEL_6;
  }
LABEL_48:
  *((_DWORD *)a1 + 2) = (2 * (*((_DWORD *)a1 + 2) >> 1) + 2) | a1[1] & 1;
  BUG();
}
