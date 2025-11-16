// Function: sub_1FEDBC0
// Address: 0x1fedbc0
//
__int64 __fastcall sub_1FEDBC0(
        __int64 a1,
        __int64 a2,
        __m128i a3,
        double a4,
        __m128i a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9)
{
  char *v10; // rdx
  char v11; // al
  __int64 v12; // rdx
  char v13; // al
  __int64 v14; // rdx
  __int64 v15; // rsi
  _QWORD *v16; // rdi
  unsigned __int64 v17; // rdx
  __int64 v18; // rbx
  __int128 v19; // kr10_16
  unsigned int v20; // eax
  __int64 v21; // rax
  __int64 v22; // rbx
  char v23; // di
  _QWORD *v24; // rdi
  __int64 v25; // rax
  int v26; // r8d
  int v27; // r9d
  __int64 v28; // rdx
  __int64 v29; // r13
  __int64 v30; // r12
  __int64 v31; // rdx
  __int64 *v32; // rdx
  __int64 v33; // rcx
  __int64 v34; // rax
  __int64 v35; // r12
  __int64 v36; // rcx
  __int64 v37; // rdx
  __int64 v38; // r13
  __int64 v39; // rax
  __int64 *v40; // r12
  unsigned int v41; // edx
  unsigned __int64 v42; // r13
  __int64 *v43; // rdx
  __int64 v44; // rax
  char v45; // r8
  __int64 v46; // rcx
  bool v47; // al
  unsigned int v48; // eax
  char v49; // r8
  unsigned int v50; // eax
  _QWORD *v51; // rdi
  __int64 v52; // rax
  __int64 v53; // rdx
  __int64 v54; // rax
  unsigned int v55; // eax
  __int64 v56; // rax
  unsigned int v57; // eax
  __int64 *v58; // rdi
  __int64 v59; // r9
  __int64 v60; // r8
  __int64 v61; // r12
  __int64 v63; // rdx
  __int64 *v64; // rax
  unsigned int v65; // edx
  __int128 v66; // [rsp-20h] [rbp-280h]
  __int128 v67; // [rsp-10h] [rbp-270h]
  int v68; // [rsp+20h] [rbp-240h]
  char v69; // [rsp+26h] [rbp-23Ah]
  char v70; // [rsp+27h] [rbp-239h]
  __int64 *v71; // [rsp+28h] [rbp-238h]
  __int64 v72; // [rsp+30h] [rbp-230h]
  __int64 *v73; // [rsp+30h] [rbp-230h]
  char v74; // [rsp+38h] [rbp-228h]
  __int64 *v75; // [rsp+38h] [rbp-228h]
  unsigned int v76; // [rsp+40h] [rbp-220h]
  __int64 v77; // [rsp+48h] [rbp-218h]
  __int64 v78; // [rsp+58h] [rbp-208h]
  __int64 v79; // [rsp+60h] [rbp-200h]
  _DWORD *v80; // [rsp+68h] [rbp-1F8h]
  __int64 v81; // [rsp+78h] [rbp-1E8h]
  __int64 v82; // [rsp+80h] [rbp-1E0h]
  __int64 v83; // [rsp+88h] [rbp-1D8h]
  __int64 v84; // [rsp+90h] [rbp-1D0h]
  unsigned int v85; // [rsp+A0h] [rbp-1C0h]
  unsigned int v86; // [rsp+A4h] [rbp-1BCh]
  _QWORD *v88; // [rsp+B0h] [rbp-1B0h]
  unsigned __int64 v89; // [rsp+B8h] [rbp-1A8h]
  unsigned int v90; // [rsp+E0h] [rbp-180h] BYREF
  __int64 v91; // [rsp+E8h] [rbp-178h]
  __int64 v92; // [rsp+F0h] [rbp-170h] BYREF
  __int64 v93; // [rsp+F8h] [rbp-168h]
  __int64 v94; // [rsp+100h] [rbp-160h] BYREF
  int v95; // [rsp+108h] [rbp-158h]
  char v96[8]; // [rsp+110h] [rbp-150h] BYREF
  __int64 v97; // [rsp+118h] [rbp-148h]
  __int128 v98; // [rsp+120h] [rbp-140h] BYREF
  __int64 v99; // [rsp+130h] [rbp-130h]
  __int64 v100; // [rsp+140h] [rbp-120h] BYREF
  __int64 v101; // [rsp+148h] [rbp-118h]
  __int64 v102; // [rsp+150h] [rbp-110h]
  __int128 v103; // [rsp+160h] [rbp-100h]
  __int64 v104; // [rsp+170h] [rbp-F0h]
  __int128 v105; // [rsp+180h] [rbp-E0h]
  __int64 v106; // [rsp+190h] [rbp-D0h]
  _BYTE *v107; // [rsp+1A0h] [rbp-C0h] BYREF
  __int64 v108; // [rsp+1A8h] [rbp-B8h]
  _BYTE v109[176]; // [rsp+1B0h] [rbp-B0h] BYREF

  v10 = *(char **)(a2 + 40);
  v11 = *v10;
  v12 = *((_QWORD *)v10 + 1);
  LOBYTE(v90) = v11;
  v91 = v12;
  if ( v11 )
  {
    switch ( v11 )
    {
      case 14:
      case 15:
      case 16:
      case 17:
      case 18:
      case 19:
      case 20:
      case 21:
      case 22:
      case 23:
      case 56:
      case 57:
      case 58:
      case 59:
      case 60:
      case 61:
        v13 = 2;
        v14 = 0;
        break;
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
        v14 = 0;
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
        v14 = 0;
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
        v14 = 0;
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
        v14 = 0;
        break;
      case 55:
        v13 = 7;
        v14 = 0;
        break;
      case 86:
      case 87:
      case 88:
      case 98:
      case 99:
      case 100:
        v13 = 8;
        v14 = 0;
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
        v14 = 0;
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
        v14 = 0;
        break;
    }
  }
  else
  {
    v13 = sub_1F596B0((__int64)&v90);
  }
  LOBYTE(v92) = v13;
  v93 = v14;
  v15 = *(_QWORD *)(a2 + 72);
  v94 = v15;
  if ( v15 )
    sub_1623A60((__int64)&v94, v15, 2);
  v16 = *(_QWORD **)(a1 + 16);
  v95 = *(_DWORD *)(a2 + 64);
  v88 = sub_1D29C20(v16, v90, v91, 1, a8, a9);
  v18 = (unsigned int)v17;
  v89 = v17;
  sub_1E341E0((__int64)&v98, *(_QWORD *)(*(_QWORD *)(a1 + 16) + 32LL), *((_DWORD *)v88 + 21), 0);
  v19 = v98;
  v70 = v99;
  v68 = HIDWORD(v99);
  v107 = v109;
  v108 = 0x800000000LL;
  if ( (_BYTE)v92 )
    v20 = sub_1FEB8F0(v92);
  else
    v20 = sub_1F58D40((__int64)&v92);
  v85 = v20 >> 3;
  v21 = *(unsigned int *)(a2 + 56);
  if ( (_DWORD)v21 )
  {
    v77 = v18;
    v84 = 16 * v18;
    v81 = 40 * v21;
    v80 = (_DWORD *)(v19 & 0xFFFFFFFFFFFFFFF8LL);
    v69 = ((__int64)v19 >> 2) & 1;
    v22 = 0;
    v86 = 0;
    while ( 1 )
    {
      if ( *(_WORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 32) + v22) + 24LL) == 48 )
        goto LABEL_17;
      v33 = v82;
      v34 = v88[5] + v84;
      LOBYTE(v33) = *(_BYTE *)v34;
      v82 = v33;
      v35 = sub_1D38BB0(*(_QWORD *)(a1 + 16), v86, (__int64)&v94, v33, *(const void ***)(v34 + 8), 0, a3, a4, a5, 0);
      v36 = v83;
      v38 = v37;
      v39 = v88[5] + v84;
      LOBYTE(v36) = *(_BYTE *)v39;
      *((_QWORD *)&v66 + 1) = v37;
      *(_QWORD *)&v66 = v35;
      v83 = v36;
      v89 = v77 | v89 & 0xFFFFFFFF00000000LL;
      v40 = sub_1D332F0(
              *(__int64 **)(a1 + 16),
              52,
              (__int64)&v94,
              v36,
              *(const void ***)(v39 + 8),
              0,
              *(double *)a3.m128i_i64,
              a4,
              a5,
              (__int64)v88,
              v89,
              v66);
      v42 = v41 | v38 & 0xFFFFFFFF00000000LL;
      v43 = (__int64 *)(v22 + *(_QWORD *)(a2 + 32));
      v44 = *(_QWORD *)(*v43 + 40) + 16LL * *((unsigned int *)v43 + 2);
      v45 = *(_BYTE *)v44;
      v46 = *(_QWORD *)(v44 + 8);
      v96[0] = v45;
      v97 = v46;
      if ( v45 )
      {
        if ( (unsigned __int8)(v45 - 14) <= 0x5Fu )
        {
          switch ( v45 )
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
              v45 = 3;
              v46 = 0;
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
              v45 = 4;
              v46 = 0;
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
              v45 = 5;
              v46 = 0;
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
              v45 = 6;
              v46 = 0;
              break;
            case 55:
              v45 = 7;
              v46 = 0;
              break;
            case 86:
            case 87:
            case 88:
            case 98:
            case 99:
            case 100:
              v45 = 8;
              v46 = 0;
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
              v45 = 9;
              v46 = 0;
              break;
            case 94:
            case 95:
            case 96:
            case 97:
            case 106:
            case 107:
            case 108:
            case 109:
              v45 = 10;
              v46 = 0;
              break;
            default:
              v45 = 2;
              v46 = 0;
              break;
          }
        }
      }
      else
      {
        v71 = v43;
        v72 = v46;
        v47 = sub_1F58D20((__int64)v96);
        v46 = v72;
        v43 = v71;
        if ( !v47 )
        {
          v23 = v92;
          LOBYTE(v100) = 0;
          v101 = v72;
          if ( !(_BYTE)v92 )
            goto LABEL_22;
          goto LABEL_31;
        }
        v45 = sub_1F596B0((__int64)v96);
        v46 = v63;
        v43 = (__int64 *)(v22 + *(_QWORD *)(a2 + 32));
      }
      v23 = v92;
      LOBYTE(v100) = v45;
      v101 = v46;
      if ( v45 == (_BYTE)v92 )
      {
        if ( v45 )
          goto LABEL_13;
LABEL_22:
        v45 = 0;
        if ( v93 == v46 )
          goto LABEL_13;
LABEL_23:
        v73 = v43;
        v74 = v45;
        v48 = sub_1F58D40((__int64)&v92);
        v49 = v74;
        v76 = v48;
        v75 = v73;
        if ( v49 )
          goto LABEL_24;
        goto LABEL_32;
      }
      if ( !(_BYTE)v92 )
        goto LABEL_23;
LABEL_31:
      v75 = v43;
      v76 = sub_1FEB8F0(v23);
      if ( v49 )
      {
LABEL_24:
        v50 = sub_1FEB8F0(v49);
        v43 = v75;
        goto LABEL_25;
      }
LABEL_32:
      v50 = sub_1F58D40((__int64)&v100);
      v43 = v75;
LABEL_25:
      if ( v50 > v76 )
      {
        v51 = *(_QWORD **)(a1 + 16);
        v100 = 0;
        v101 = 0;
        v102 = 0;
        if ( v80 )
        {
          if ( v69 )
          {
            *((_QWORD *)&v103 + 1) = *((_QWORD *)&v19 + 1) + v86;
            LOBYTE(v104) = v70;
            *(_QWORD *)&v103 = v19 & 0xFFFFFFFFFFFFFFF8LL | 4;
            HIDWORD(v104) = v80[3];
          }
          else
          {
            *((_QWORD *)&v103 + 1) = *((_QWORD *)&v19 + 1) + v86;
            *(_QWORD *)&v103 = v19 & 0xFFFFFFFFFFFFFFF8LL;
            LOBYTE(v104) = v70;
            v54 = *(_QWORD *)v80;
            if ( *(_BYTE *)(*(_QWORD *)v80 + 8LL) == 16 )
              v55 = *(_DWORD *)(**(_QWORD **)(v54 + 16) + 8LL);
            else
              v55 = *(_DWORD *)(v54 + 8);
            HIDWORD(v104) = v55 >> 8;
          }
        }
        else
        {
          LODWORD(v104) = 0;
          v103 = 0u;
          HIDWORD(v104) = v68;
        }
        v78 &= 0xFFFFFFFF00000000LL;
        v52 = sub_1D2C750(
                v51,
                (__int64)(v51 + 11),
                v78,
                (__int64)&v94,
                *v43,
                v43[1],
                (__int64)v40,
                v42,
                v103,
                v104,
                v92,
                v93,
                0,
                0,
                (__int64)&v100);
        v29 = v53;
        v30 = v52;
        v31 = (unsigned int)v108;
        if ( (unsigned int)v108 < HIDWORD(v108) )
          goto LABEL_16;
        goto LABEL_29;
      }
LABEL_13:
      v24 = *(_QWORD **)(a1 + 16);
      v100 = 0;
      v101 = 0;
      v102 = 0;
      if ( v80 )
      {
        if ( v69 )
        {
          *((_QWORD *)&v105 + 1) = *((_QWORD *)&v19 + 1) + v86;
          LOBYTE(v106) = v70;
          *(_QWORD *)&v105 = v19 & 0xFFFFFFFFFFFFFFF8LL | 4;
          HIDWORD(v106) = v80[3];
        }
        else
        {
          *((_QWORD *)&v105 + 1) = *((_QWORD *)&v19 + 1) + v86;
          *(_QWORD *)&v105 = v19 & 0xFFFFFFFFFFFFFFF8LL;
          LOBYTE(v106) = v70;
          v56 = *(_QWORD *)v80;
          if ( *(_BYTE *)(*(_QWORD *)v80 + 8LL) == 16 )
            v57 = *(_DWORD *)(**(_QWORD **)(v56 + 16) + 8LL);
          else
            v57 = *(_DWORD *)(v56 + 8);
          HIDWORD(v106) = v57 >> 8;
        }
      }
      else
      {
        LODWORD(v106) = 0;
        v105 = 0u;
        HIDWORD(v106) = v68;
      }
      v79 &= 0xFFFFFFFF00000000LL;
      v25 = sub_1D2BF40(
              v24,
              (__int64)(v24 + 11),
              v79,
              (__int64)&v94,
              *v43,
              v43[1],
              (__int64)v40,
              v42,
              v105,
              v106,
              0,
              0,
              (__int64)&v100);
      v29 = v28;
      v30 = v25;
      v31 = (unsigned int)v108;
      if ( (unsigned int)v108 < HIDWORD(v108) )
        goto LABEL_16;
LABEL_29:
      sub_16CD150((__int64)&v107, v109, 0, 16, v26, v27);
      v31 = (unsigned int)v108;
LABEL_16:
      v32 = (__int64 *)&v107[16 * v31];
      *v32 = v30;
      v32[1] = v29;
      LODWORD(v108) = v108 + 1;
LABEL_17:
      v22 += 40;
      v86 += v85;
      if ( v81 == v22 )
      {
        v58 = *(__int64 **)(a1 + 16);
        v59 = 0;
        if ( !(_DWORD)v108 )
          goto LABEL_42;
        *((_QWORD *)&v67 + 1) = (unsigned int)v108;
        *(_QWORD *)&v67 = v107;
        v64 = sub_1D359D0(v58, 2, (__int64)&v94, 1, 0, 0, *(double *)a3.m128i_i64, a4, a5, v67);
        v58 = *(__int64 **)(a1 + 16);
        v60 = (__int64)v64;
        v59 = v65;
        goto LABEL_43;
      }
    }
  }
  v58 = *(__int64 **)(a1 + 16);
  v59 = 0;
  v77 = (unsigned int)v18;
LABEL_42:
  v60 = (__int64)(v58 + 11);
LABEL_43:
  v100 = 0;
  v101 = 0;
  v102 = 0;
  v61 = sub_1D2B730(
          v58,
          v90,
          v91,
          (__int64)&v94,
          v60,
          v59,
          (__int64)v88,
          v77 | v89 & 0xFFFFFFFF00000000LL,
          v98,
          v99,
          0,
          0,
          (__int64)&v100,
          0);
  if ( v107 != v109 )
    _libc_free((unsigned __int64)v107);
  if ( v94 )
    sub_161E7C0((__int64)&v94, v94);
  return v61;
}
