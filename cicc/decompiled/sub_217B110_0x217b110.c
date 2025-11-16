// Function: sub_217B110
// Address: 0x217b110
//
__int64 __fastcall sub_217B110(__int64 a1, __int64 a2, __m128i a3, double a4, __m128i a5)
{
  _QWORD *v7; // r15
  char v8; // dl
  __int64 v9; // rax
  char v10; // al
  __int64 v11; // rdx
  __int64 v12; // r8
  char v13; // dl
  bool v14; // al
  unsigned int v15; // r13d
  __int64 v16; // rdx
  __int64 v17; // rcx
  int v18; // r8d
  int v19; // r9d
  unsigned __int8 v20; // al
  __int64 v21; // r14
  __int64 v22; // rsi
  __int16 v23; // ax
  __int64 v24; // rdx
  __int64 *v25; // rax
  _BOOL4 v26; // r13d
  __int64 v27; // r14
  unsigned int v28; // edx
  unsigned int v29; // r13d
  char *v30; // rax
  unsigned __int8 v31; // dl
  const void **v32; // r15
  __int64 v33; // rax
  __int64 v35; // r13
  __int64 v36; // rax
  unsigned int v37; // r14d
  __int64 v38; // rdx
  __int64 v39; // rcx
  int v40; // r8d
  int v41; // r9d
  int v42; // r9d
  __int64 v43; // rdx
  int v44; // r14d
  __int64 v45; // rax
  unsigned int v46; // r13d
  const __m128i *v47; // r8
  __int64 v48; // rax
  __int64 v49; // r13
  int v50; // edx
  __int64 v51; // rax
  __int64 v52; // rcx
  __int64 v53; // rdx
  __int64 v54; // r9
  __int64 *v55; // rax
  __int64 v56; // r14
  unsigned int v57; // edx
  __int64 v58; // rsi
  __int64 v59; // rcx
  __int64 *v60; // r10
  unsigned int v61; // r12d
  __int64 *v62; // rdi
  unsigned int v63; // r13d
  __int64 v64; // rdx
  __int64 v65; // rcx
  int v66; // r8d
  int v67; // r9d
  __int64 v68; // rdx
  int v69; // r9d
  int v70; // r14d
  __int64 v71; // rax
  unsigned int v72; // r13d
  const __m128i *v73; // r8
  __int64 v74; // rax
  __int64 v75; // r13
  int v76; // edx
  __int64 v77; // rax
  __int64 v78; // rcx
  __int64 v79; // rdx
  __int64 v80; // r9
  __int64 *v81; // rax
  __int64 v82; // r14
  unsigned int v83; // edx
  __int64 v84; // [rsp+0h] [rbp-170h]
  __int64 v85; // [rsp+0h] [rbp-170h]
  __int64 v86; // [rsp+8h] [rbp-168h]
  __int64 v87; // [rsp+8h] [rbp-168h]
  __int64 v88; // [rsp+18h] [rbp-158h]
  __int64 v89; // [rsp+18h] [rbp-158h]
  __int64 v90; // [rsp+20h] [rbp-150h]
  __int64 v91; // [rsp+20h] [rbp-150h]
  int v92; // [rsp+28h] [rbp-148h]
  int v93; // [rsp+28h] [rbp-148h]
  __int64 v94; // [rsp+28h] [rbp-148h]
  __int64 v95; // [rsp+28h] [rbp-148h]
  __int64 *v96; // [rsp+40h] [rbp-130h]
  __int64 v97; // [rsp+48h] [rbp-128h]
  __int64 *v98; // [rsp+50h] [rbp-120h]
  __int64 v99; // [rsp+60h] [rbp-110h] BYREF
  __int64 v100; // [rsp+68h] [rbp-108h] BYREF
  char v101[8]; // [rsp+70h] [rbp-100h] BYREF
  __int64 v102; // [rsp+78h] [rbp-F8h]
  __int64 v103; // [rsp+80h] [rbp-F0h] BYREF
  int v104; // [rsp+88h] [rbp-E8h]
  __m128i v105; // [rsp+90h] [rbp-E0h] BYREF
  __int64 v106; // [rsp+A0h] [rbp-D0h]
  __int64 *v107; // [rsp+B0h] [rbp-C0h] BYREF
  __int64 v108; // [rsp+B8h] [rbp-B8h]
  _QWORD v109[22]; // [rsp+C0h] [rbp-B0h] BYREF

  v7 = *(_QWORD **)(a2 + 16);
  v99 = 0;
  v100 = 0;
  if ( !sub_2177670(a1, 0, &v99, &v100) || !v99 )
    return 0;
  v8 = *(_BYTE *)(v99 + 88);
  v9 = *(_QWORD *)(v99 + 96);
  v101[0] = v8;
  v102 = v9;
  if ( v8 )
  {
    if ( (unsigned __int8)(v8 - 14) > 0x5Fu )
      return 0;
    switch ( v8 )
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
        v13 = 3;
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
        v13 = 4;
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
        v13 = 5;
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
        v13 = 6;
        break;
      case 55:
        v13 = 7;
        break;
      case 86:
      case 87:
      case 88:
      case 98:
      case 99:
      case 100:
        v13 = 8;
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
        v13 = 9;
        break;
      case 94:
      case 95:
      case 96:
      case 97:
      case 106:
      case 107:
      case 108:
      case 109:
        v13 = 10;
        break;
      default:
        v13 = 2;
        break;
    }
LABEL_35:
    v14 = (unsigned __int8)(v13 - 14) <= 0x47u || (unsigned __int8)(v13 - 2) <= 5u;
    goto LABEL_7;
  }
  if ( !sub_1F58D20((__int64)v101) )
    return 0;
  v10 = sub_1F596B0((__int64)v101);
  v12 = v11;
  LOBYTE(v107) = v10;
  v13 = v10;
  v108 = v12;
  if ( v10 )
    goto LABEL_35;
  v14 = sub_1F58CF0((__int64)&v107);
LABEL_7:
  if ( v14 )
  {
    v15 = v101[0] ? sub_216FFF0(v101[0]) : sub_1F58D40((__int64)v101);
    if ( v15 == v100 && (((v15 - 16) & 0xFFFFFFEF) == 0 || v15 == 64) && sub_1D18C00(v99, 1, 0) )
    {
      if ( v15 == 32 )
      {
        v20 = 5;
      }
      else if ( v15 > 0x20 )
      {
        v20 = 6;
        if ( v15 != 64 )
        {
          v20 = 0;
          v16 = 7;
          if ( v15 == 128 )
            v20 = 7;
        }
      }
      else
      {
        v20 = 3;
        if ( v15 != 8 )
        {
          v20 = 4;
          if ( v15 != 16 )
            v20 = 2 * (v15 == 1);
        }
      }
      v21 = v99;
      v22 = v20;
      v23 = *(_WORD *)(v99 + 24);
      if ( v23 > 662 )
      {
        if ( (unsigned __int16)(v23 - 663) > 1u )
          return 0;
        v107 = v109;
        v108 = 0x800000000LL;
        sub_1D23890((__int64)&v107, *(const __m128i **)(v99 + 32), v16, v17, v18, v19);
        v35 = *(_QWORD *)(a2 + 16);
        v36 = sub_1E0A0C0(v7[4]);
        v37 = (unsigned __int8)sub_21700D0(v36, 0);
        v103 = *(_QWORD *)(a1 + 72);
        if ( v103 )
          sub_21700C0(&v103);
        v104 = *(_DWORD *)(a1 + 64);
        v105.m128i_i64[0] = sub_1D38BB0(v35, 4070, (__int64)&v103, v37, 0, 0, a3, a4, a5, 0);
        v105.m128i_i64[1] = v38;
        sub_1D23890((__int64)&v107, &v105, v38, v39, v40, v41);
        sub_17CD270(&v103);
        v43 = v99;
        v44 = *(_DWORD *)(v99 + 56);
        if ( v44 != 1 )
        {
          v45 = (unsigned int)v108;
          v46 = 1;
          while ( 1 )
          {
            v47 = (const __m128i *)(*(_QWORD *)(v43 + 32) + 40LL * v46);
            if ( HIDWORD(v108) <= (unsigned int)v45 )
            {
              v95 = *(_QWORD *)(v43 + 32) + 40LL * v46;
              sub_16CD150((__int64)&v107, v109, 0, 16, (int)v47, v42);
              v45 = (unsigned int)v108;
              v47 = (const __m128i *)v95;
            }
            a3 = _mm_loadu_si128(v47);
            ++v46;
            *(__m128i *)&v107[2 * v45] = a3;
            v45 = (unsigned int)(v108 + 1);
            LODWORD(v108) = v108 + 1;
            if ( v46 == v44 )
              break;
            v43 = v99;
          }
        }
        v48 = sub_1D252B0(*(_QWORD *)(a2 + 16), v22, 0, 1, 0);
        v105 = 0u;
        v49 = v48;
        v106 = 0;
        v92 = v50;
        v51 = sub_1E34390(*(_QWORD *)(v99 + 104));
        v52 = v99;
        v53 = (unsigned int)v108;
        v54 = v51;
        v55 = v107;
        v56 = *(_QWORD *)(v99 + 104);
        v103 = *(_QWORD *)(v99 + 72);
        if ( v103 )
        {
          v84 = (__int64)v107;
          v86 = (unsigned int)v108;
          v88 = v54;
          v90 = v99;
          sub_21700C0(&v103);
          v55 = (__int64 *)v84;
          v53 = v86;
          v54 = v88;
          v52 = v90;
        }
        v104 = *(_DWORD *)(v52 + 64);
        v27 = sub_1D251C0(
                v7,
                44,
                (__int64)&v103,
                v49,
                v92,
                v54,
                v55,
                v53,
                v22,
                0,
                *(_OWORD *)v56,
                *(_QWORD *)(v56 + 16),
                3u,
                0,
                (__int64)&v105);
        v29 = v57;
        sub_17CD270(&v103);
        v58 = v99;
        v59 = v27;
        if ( *(_WORD *)(v99 + 24) == 663 )
        {
LABEL_80:
          sub_1D44C70((__int64)v7, v58, 2, v59, 1u);
LABEL_62:
          if ( v107 != v109 )
            _libc_free((unsigned __int64)v107);
          goto LABEL_26;
        }
      }
      else
      {
        if ( v23 <= 660 )
        {
          if ( (unsigned __int16)(v23 - 659) <= 1u )
          {
            v107 = 0;
            v108 = 0;
            v109[0] = 0;
            v24 = *(_QWORD *)(v99 + 104);
            v25 = *(__int64 **)(v99 + 32);
            v26 = (*(_BYTE *)(v99 + 26) & 8) != 0;
            v105.m128i_i64[0] = *(_QWORD *)(v99 + 72);
            if ( v105.m128i_i64[0] )
            {
              v96 = v25;
              v97 = v24;
              sub_21700C0(v105.m128i_i64);
              v25 = v96;
              v24 = v97;
            }
            v105.m128i_i32[2] = *(_DWORD *)(v21 + 64);
            v27 = sub_1D2B730(
                    v7,
                    v22,
                    0,
                    (__int64)&v105,
                    *v25,
                    v25[1],
                    v25[5],
                    v25[6],
                    *(_OWORD *)v24,
                    *(_QWORD *)(v24 + 16),
                    v26,
                    0,
                    (__int64)&v107,
                    0);
            v29 = v28;
            sub_17CD270(v105.m128i_i64);
            if ( *(_WORD *)(v99 + 24) == 659 )
              sub_1D44C70((__int64)v7, v99, 2, v27, 1u);
            else
              sub_1D44C70((__int64)v7, v99, 4, v27, 1u);
LABEL_26:
            if ( v27 )
            {
              v30 = *(char **)(a1 + 40);
              v31 = *v30;
              v32 = (const void **)*((_QWORD *)v30 + 1);
              v33 = *(_QWORD *)(v27 + 40) + 16LL * v29;
              if ( *(_BYTE *)v33 != v31 || *(const void ***)(v33 + 8) != v32 && !v31 )
              {
                v60 = *(__int64 **)(a2 + 16);
                v61 = v31;
                v107 = *(__int64 **)(a1 + 72);
                if ( v107 )
                {
                  v98 = v60;
                  sub_21700C0((__int64 *)&v107);
                  v60 = v98;
                }
                LODWORD(v108) = *(_DWORD *)(a1 + 64);
                v27 = sub_1D321C0(
                        v60,
                        v27,
                        v29,
                        (__int64)&v107,
                        v61,
                        v32,
                        *(double *)a3.m128i_i64,
                        a4,
                        *(double *)a5.m128i_i64);
                if ( v107 )
                  sub_161E7C0((__int64)&v107, (__int64)v107);
              }
            }
            return v27;
          }
          return 0;
        }
        v107 = v109;
        v108 = 0x800000000LL;
        sub_1D23890((__int64)&v107, *(const __m128i **)(v99 + 32), v16, v17, v18, v19);
        v62 = (__int64 *)sub_1E0A0C0(v7[4]);
        v63 = (unsigned __int8)sub_21700D0((__int64)v62, 0);
        v103 = *(_QWORD *)(a1 + 72);
        if ( v103 )
        {
          v62 = &v103;
          sub_21700C0(&v103);
        }
        v104 = *(_DWORD *)(a1 + 64);
        v105.m128i_i64[0] = sub_1D38BB0((__int64)v7, 4061, (__int64)&v103, v63, 0, 0, a3, a4, a5, 0);
        v105.m128i_i64[1] = v64;
        sub_1D23890((__int64)&v107, &v105, v64, v65, v66, v67);
        sub_17CD270(&v103);
        v68 = v99;
        v69 = (int)v62;
        v70 = *(_DWORD *)(v99 + 56);
        if ( v70 != 1 )
        {
          v71 = (unsigned int)v108;
          v72 = 1;
          while ( 1 )
          {
            v73 = (const __m128i *)(*(_QWORD *)(v68 + 32) + 40LL * v72);
            if ( (unsigned int)v71 >= HIDWORD(v108) )
            {
              v94 = *(_QWORD *)(v68 + 32) + 40LL * v72;
              sub_16CD150((__int64)&v107, v109, 0, 16, (int)v73, v69);
              v71 = (unsigned int)v108;
              v73 = (const __m128i *)v94;
            }
            a3 = _mm_loadu_si128(v73);
            ++v72;
            *(__m128i *)&v107[2 * v71] = a3;
            v71 = (unsigned int)(v108 + 1);
            LODWORD(v108) = v108 + 1;
            if ( v72 == v70 )
              break;
            v68 = v99;
          }
        }
        v74 = sub_1D252B0((__int64)v7, v22, 0, 1, 0);
        v105 = 0u;
        v75 = v74;
        v106 = 0;
        v93 = v76;
        v77 = sub_1E34390(*(_QWORD *)(v99 + 104));
        v78 = v99;
        v79 = (unsigned int)v108;
        v80 = v77;
        v81 = v107;
        v82 = *(_QWORD *)(v99 + 104);
        v103 = *(_QWORD *)(v99 + 72);
        if ( v103 )
        {
          v85 = (__int64)v107;
          v87 = (unsigned int)v108;
          v89 = v80;
          v91 = v99;
          sub_21700C0(&v103);
          v81 = (__int64 *)v85;
          v79 = v87;
          v80 = v89;
          v78 = v91;
        }
        v104 = *(_DWORD *)(v78 + 64);
        v27 = sub_1D251C0(
                v7,
                44,
                (__int64)&v103,
                v75,
                v93,
                v80,
                v81,
                v79,
                v22,
                0,
                *(_OWORD *)v82,
                *(_QWORD *)(v82 + 16),
                3u,
                0,
                (__int64)&v105);
        v29 = v83;
        sub_17CD270(&v103);
        v58 = v99;
        v59 = v27;
        if ( *(_WORD *)(v99 + 24) == 661 )
          goto LABEL_80;
      }
      sub_1D44C70((__int64)v7, v58, 4, v59, 1u);
      goto LABEL_62;
    }
  }
  return 0;
}
