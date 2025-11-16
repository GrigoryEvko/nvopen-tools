// Function: sub_201B410
// Address: 0x201b410
//
unsigned __int64 __fastcall sub_201B410(
        __int64 **a1,
        unsigned __int64 a2,
        unsigned int a3,
        __m128i a4,
        double a5,
        __m128i a6)
{
  unsigned __int64 v6; // r12
  __int64 v7; // rax
  char v8; // dl
  const void **v9; // rax
  __int64 v10; // rax
  const void **v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r15
  unsigned int v14; // ebx
  unsigned int v15; // eax
  unsigned int v16; // r14d
  unsigned int v17; // eax
  __int64 *v18; // r13
  const __m128i *v19; // rax
  __int64 v20; // rsi
  __int64 v21; // rsi
  __m128i v22; // xmm1
  __int64 v23; // rdi
  __int64 v24; // rax
  unsigned int v25; // edx
  char v26; // al
  unsigned int v27; // esi
  int v28; // r8d
  __int64 v29; // r9
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rax
  unsigned __int64 v33; // rax
  __int64 v34; // r12
  unsigned __int64 v35; // r13
  __int64 v36; // rax
  __int64 *m128i_i64; // rax
  __int64 v38; // rax
  unsigned __int64 v39; // r15
  __int64 v40; // r14
  __int64 v41; // rax
  char v42; // dl
  __int64 v43; // rax
  __int64 v44; // rdx
  __int64 *v45; // rdi
  unsigned int v46; // esi
  int v47; // r8d
  int v48; // r9d
  __int64 *v49; // r13
  __int64 v50; // rax
  __int64 **v51; // rax
  __int64 v52; // rax
  __int64 **v53; // rax
  __m128i *v54; // rdi
  _BYTE *v55; // rsi
  __int64 v56; // rcx
  __int64 v57; // r15
  __int64 *v58; // r14
  __int64 v59; // rdx
  __int64 v60; // rbx
  __int64 *v61; // rax
  __int64 v62; // r8
  __int64 **v63; // rbx
  unsigned __int64 v64; // r15
  __int64 v65; // rdx
  __int64 v66; // r13
  __int128 v68; // [rsp-20h] [rbp-5C0h]
  __int128 v69; // [rsp-10h] [rbp-5B0h]
  __int128 v70; // [rsp-10h] [rbp-5B0h]
  __m128i v71; // [rsp+20h] [rbp-580h] BYREF
  __int64 v72; // [rsp+30h] [rbp-570h]
  __int64 v73; // [rsp+38h] [rbp-568h]
  __int64 v74; // [rsp+40h] [rbp-560h]
  const void **v75; // [rsp+48h] [rbp-558h]
  unsigned int v76; // [rsp+50h] [rbp-550h]
  unsigned __int8 v77; // [rsp+57h] [rbp-549h]
  __m128i *v78; // [rsp+58h] [rbp-548h]
  __int128 v79; // [rsp+60h] [rbp-540h]
  __m128i *v80; // [rsp+70h] [rbp-530h]
  __int64 v81; // [rsp+78h] [rbp-528h]
  __int64 v82; // [rsp+80h] [rbp-520h]
  __int64 *v83; // [rsp+88h] [rbp-518h]
  __int64 **v84; // [rsp+90h] [rbp-510h]
  __int64 v85; // [rsp+98h] [rbp-508h]
  __int64 *v86; // [rsp+A0h] [rbp-500h]
  __int64 v87; // [rsp+A8h] [rbp-4F8h]
  __int64 v88; // [rsp+B0h] [rbp-4F0h] BYREF
  const void **v89; // [rsp+B8h] [rbp-4E8h]
  __int64 v90; // [rsp+C0h] [rbp-4E0h] BYREF
  int v91; // [rsp+C8h] [rbp-4D8h]
  char v92[8]; // [rsp+D0h] [rbp-4D0h] BYREF
  __int64 v93; // [rsp+D8h] [rbp-4C8h]
  _QWORD v94[2]; // [rsp+E0h] [rbp-4C0h] BYREF
  char v95; // [rsp+F0h] [rbp-4B0h]
  __int64 v96; // [rsp+F8h] [rbp-4A8h]
  __m128i *v97; // [rsp+100h] [rbp-4A0h] BYREF
  __int64 v98; // [rsp+108h] [rbp-498h]
  char v99; // [rsp+110h] [rbp-490h] BYREF
  _BYTE *v100; // [rsp+150h] [rbp-450h] BYREF
  __int64 v101; // [rsp+158h] [rbp-448h]
  _BYTE v102[512]; // [rsp+160h] [rbp-440h] BYREF
  _BYTE *v103; // [rsp+360h] [rbp-240h] BYREF
  __int64 v104; // [rsp+368h] [rbp-238h]
  _BYTE v105[560]; // [rsp+370h] [rbp-230h] BYREF

  v6 = a2;
  v7 = *(_QWORD *)(a2 + 40) + 16LL * a3;
  v84 = a1;
  v8 = *(_BYTE *)v7;
  v9 = *(const void ***)(v7 + 8);
  LOBYTE(v88) = v8;
  v89 = v9;
  if ( v8 )
  {
    switch ( v8 )
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
        v77 = 2;
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
        v77 = 3;
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
        v77 = 4;
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
        v77 = 5;
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
        v77 = 6;
        break;
      case 55:
        v77 = 7;
        break;
      case 86:
      case 87:
      case 88:
      case 98:
      case 99:
      case 100:
        v77 = 8;
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
        v77 = 9;
        break;
      case 94:
      case 95:
      case 96:
      case 97:
      case 106:
      case 107:
      case 108:
      case 109:
        v77 = 10;
        break;
    }
    v75 = 0;
    v12 = 0;
    v13 = v77;
    v14 = v77;
  }
  else
  {
    LOBYTE(v10) = sub_1F596B0((__int64)&v88);
    v75 = v11;
    v12 = (__int64)v11;
    v8 = v88;
    v13 = v10;
    v77 = v10;
    v14 = v10;
    if ( !(_BYTE)v88 )
    {
      v85 = v12;
      v15 = sub_1F58D30((__int64)&v88);
      v12 = v85;
      v16 = v15;
      goto LABEL_5;
    }
  }
  v16 = word_4301260[(unsigned __int8)(v8 - 14)];
LABEL_5:
  v17 = *(_DWORD *)(a2 + 56);
  v94[0] = v13;
  v83 = &v90;
  v76 = v17;
  v18 = *v84;
  v19 = *(const __m128i **)(a2 + 32);
  v20 = (*v84)[2];
  v94[1] = v12;
  v95 = 1;
  v73 = v20;
  v21 = *(_QWORD *)(v6 + 72);
  v96 = 0;
  v22 = _mm_loadu_si128(v19);
  v90 = v21;
  v71 = v22;
  if ( v21 )
  {
    sub_1623A60((__int64)&v90, v21, 2);
    v18 = *v84;
  }
  v91 = *(_DWORD *)(v6 + 64);
  v100 = v102;
  v101 = 0x2000000000LL;
  v103 = v105;
  v104 = 0x2000000000LL;
  if ( v16 )
  {
    v82 = 0;
    v72 = v16;
    v85 = 40LL * (v76 - 2) + 80;
    v78 = (__m128i *)&v99;
    while ( 1 )
    {
      v23 = v18[4];
      v97 = v78;
      v98 = 0x400000000LL;
      *(_QWORD *)&v79 = *(_QWORD *)(*(_QWORD *)v73 + 48LL);
      v24 = sub_1E0A0C0(v23);
      if ( (__int64 (__fastcall *)(__int64, __int64))v79 == sub_1D13A20 )
      {
        v25 = 8 * sub_15A9520(v24, 0);
        if ( v25 == 32 )
        {
          v26 = 5;
        }
        else if ( v25 > 0x20 )
        {
          v26 = 6;
          if ( v25 != 64 )
          {
            v26 = 0;
            if ( v25 == 128 )
              v26 = 7;
          }
        }
        else
        {
          v26 = 3;
          if ( v25 != 8 )
            v26 = 4 * (v25 == 16);
        }
      }
      else
      {
        v26 = ((__int64 (__fastcall *)(__int64, __int64))v79)(v73, v24);
      }
      v27 = v74;
      LOBYTE(v27) = v26;
      *(_QWORD *)&v79 = sub_1D38BB0((__int64)v18, v82, (__int64)v83, v27, 0, 0, a4, *(double *)v22.m128i_i64, a6, 0);
      v30 = (unsigned int)v98;
      *((_QWORD *)&v79 + 1) = v31;
      if ( (unsigned int)v98 >= HIDWORD(v98) )
      {
        sub_16CD150((__int64)&v97, v78, 0, 16, v28, v29);
        v30 = (unsigned int)v98;
      }
      a4 = _mm_load_si128(&v71);
      v97[v30] = a4;
      v32 = (unsigned int)(v98 + 1);
      LODWORD(v98) = v98 + 1;
      if ( v76 > 1 )
        break;
LABEL_26:
      v81 = v32;
      v45 = *v84;
      v46 = *(unsigned __int16 *)(v6 + 24);
      v80 = v97;
      *((_QWORD *)&v69 + 1) = v32;
      *(_QWORD *)&v69 = v97;
      v49 = sub_1D373B0(
              v45,
              v46,
              (__int64)v83,
              (unsigned __int8 *)v94,
              2,
              *(double *)a4.m128i_i64,
              *(double *)v22.m128i_i64,
              a6,
              v29,
              v69);
      v50 = (unsigned int)v101;
      if ( (unsigned int)v101 >= HIDWORD(v101) )
      {
        sub_16CD150((__int64)&v100, v102, 0, 16, v47, v48);
        v50 = (unsigned int)v101;
      }
      v51 = (__int64 **)&v100[16 * v50];
      *v51 = v49;
      v51[1] = 0;
      v52 = (unsigned int)v104;
      LODWORD(v101) = v101 + 1;
      if ( (unsigned int)v104 >= HIDWORD(v104) )
      {
        sub_16CD150((__int64)&v103, v105, 0, 16, v47, v48);
        v52 = (unsigned int)v104;
      }
      v53 = (__int64 **)&v103[16 * v52];
      *v53 = v49;
      v54 = v97;
      v53[1] = (__int64 *)1;
      LODWORD(v104) = v104 + 1;
      if ( v54 != v78 )
        _libc_free((unsigned __int64)v54);
      ++v82;
      v18 = *v84;
      if ( v72 == v82 )
      {
        v55 = v100;
        v56 = (unsigned int)v101;
        goto LABEL_34;
      }
    }
    v33 = v6;
    v34 = 40;
    v35 = v33;
    while ( 1 )
    {
      v38 = *(_QWORD *)(v35 + 32);
      v39 = *(_QWORD *)(v38 + v34 + 8);
      v40 = *(_QWORD *)(v38 + v34);
      v41 = *(_QWORD *)(v40 + 40) + 16LL * (unsigned int)v39;
      v42 = *(_BYTE *)v41;
      v43 = *(_QWORD *)(v41 + 8);
      v92[0] = v42;
      v93 = v43;
      if ( v42 )
      {
        if ( (unsigned __int8)(v42 - 14) > 0x5Fu )
          goto LABEL_19;
      }
      else if ( !sub_1F58D20((__int64)v92) )
      {
LABEL_19:
        v36 = (unsigned int)v98;
        if ( (unsigned int)v98 >= HIDWORD(v98) )
          goto LABEL_24;
        goto LABEL_20;
      }
      LOBYTE(v14) = v77;
      v86 = sub_1D332F0(
              *v84,
              106,
              (__int64)v83,
              v14,
              v75,
              0,
              *(double *)a4.m128i_i64,
              *(double *)v22.m128i_i64,
              a6,
              v40,
              v39,
              v79);
      v40 = (__int64)v86;
      v87 = v44;
      v39 = (unsigned int)v44 | v39 & 0xFFFFFFFF00000000LL;
      v36 = (unsigned int)v98;
      if ( (unsigned int)v98 >= HIDWORD(v98) )
      {
LABEL_24:
        sub_16CD150((__int64)&v97, v78, 0, 16, v28, v29);
        v36 = (unsigned int)v98;
      }
LABEL_20:
      m128i_i64 = v97[v36].m128i_i64;
      v34 += 40;
      *m128i_i64 = v40;
      m128i_i64[1] = v39;
      v32 = (unsigned int)(v98 + 1);
      LODWORD(v98) = v98 + 1;
      if ( v85 == v34 )
      {
        v6 = v35;
        goto LABEL_26;
      }
    }
  }
  v55 = v102;
  v56 = 0;
LABEL_34:
  v57 = (__int64)v83;
  *((_QWORD *)&v70 + 1) = v56;
  *(_QWORD *)&v70 = v55;
  v58 = sub_1D359D0(v18, 104, (__int64)v83, v88, v89, 0, *(double *)a4.m128i_i64, *(double *)v22.m128i_i64, a6, v70);
  v60 = v59;
  *((_QWORD *)&v68 + 1) = (unsigned int)v104;
  *(_QWORD *)&v68 = v103;
  v61 = sub_1D359D0(*v84, 2, v57, 1, 0, 0, *(double *)a4.m128i_i64, *(double *)v22.m128i_i64, a6, v68);
  v62 = v60;
  v63 = v84;
  v64 = (unsigned __int64)v61;
  v66 = v65;
  sub_201ADC0((__int64)v84, v6, 0, (unsigned __int64)v58, v62);
  sub_201ADC0((__int64)v63, v6, 1, v64, v66);
  if ( v103 != v105 )
    _libc_free((unsigned __int64)v103);
  if ( v100 != v102 )
    _libc_free((unsigned __int64)v100);
  if ( v90 )
    sub_161E7C0((__int64)v83, v90);
  return v64;
}
