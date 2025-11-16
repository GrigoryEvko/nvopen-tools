// Function: sub_1D40890
// Address: 0x1d40890
//
__int64 *__fastcall sub_1D40890(
        __int64 *a1,
        __int64 a2,
        unsigned int a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128i a7,
        double a8,
        __m128i a9)
{
  __int64 *v9; // r14
  char *v11; // rdx
  char v12; // al
  __int64 v13; // rdx
  unsigned __int8 v14; // al
  __int64 v15; // rax
  const void **v16; // rdx
  __int64 v17; // rsi
  unsigned __int64 v18; // rbx
  unsigned __int64 v19; // r12
  _BYTE *v20; // rax
  __int64 v21; // rbx
  _BYTE *i; // rdx
  __int64 v23; // rax
  __int64 v24; // rbx
  __int64 v25; // r13
  __int64 v26; // r15
  _BYTE *v27; // rax
  const __m128i *v28; // roff
  __int64 v29; // r14
  __int64 v30; // rax
  char v31; // dl
  __int64 v32; // rax
  char v33; // al
  __int64 v34; // rax
  const void **v35; // rdx
  const void **v36; // r14
  __int64 v37; // rax
  unsigned int v38; // edx
  char v39; // al
  unsigned int v40; // esi
  __int128 v41; // rax
  __int64 *v42; // rax
  int v43; // edx
  int v44; // edi
  __int64 *v45; // rdx
  unsigned __int64 v46; // rax
  unsigned int v47; // eax
  __int64 v48; // rsi
  __int64 v49; // rbx
  __int64 v50; // rdx
  __int128 v51; // rax
  __int64 v52; // rcx
  __int64 *v53; // rax
  unsigned __int64 v54; // rdx
  unsigned __int64 v55; // r13
  __int64 v56; // rdx
  unsigned __int64 *v57; // rdx
  unsigned int v58; // r12d
  unsigned int v59; // r15d
  _QWORD *v60; // rax
  __int64 v61; // rdx
  _QWORD *v62; // r8
  __int64 v63; // rax
  _QWORD *v64; // rax
  unsigned int v65; // eax
  __int64 v66; // r13
  unsigned __int8 v67; // al
  const void **v68; // r8
  __int64 *v69; // r14
  __int64 v71; // rax
  __int64 *v72; // rax
  unsigned __int64 v73; // rdx
  __int64 v74; // rax
  __int64 v75; // rdx
  char v76; // al
  __int64 v77; // rdx
  __int64 v78; // rax
  unsigned __int64 v79; // rdx
  __int64 v80; // rbx
  __int128 v81; // rax
  __int64 v82; // rcx
  const void **v83; // rdx
  __int128 v84; // [rsp-10h] [rbp-1F0h]
  __int128 v85; // [rsp-10h] [rbp-1F0h]
  __int64 v86; // [rsp-10h] [rbp-1F0h]
  __int128 v87; // [rsp-10h] [rbp-1F0h]
  __int64 v88; // [rsp+10h] [rbp-1D0h]
  __int64 v89; // [rsp+18h] [rbp-1C8h]
  const void **v90; // [rsp+20h] [rbp-1C0h]
  unsigned int v91; // [rsp+28h] [rbp-1B8h]
  unsigned int v92; // [rsp+2Ch] [rbp-1B4h]
  __int64 v94; // [rsp+38h] [rbp-1A8h]
  unsigned int v95; // [rsp+38h] [rbp-1A8h]
  unsigned int v96; // [rsp+40h] [rbp-1A0h]
  char v97; // [rsp+4Bh] [rbp-195h]
  unsigned int v98; // [rsp+4Ch] [rbp-194h]
  unsigned int v99; // [rsp+58h] [rbp-188h]
  __int64 v100; // [rsp+58h] [rbp-188h]
  __int64 (__fastcall *v101)(__int64, __int64); // [rsp+60h] [rbp-180h]
  __int64 v102; // [rsp+68h] [rbp-178h]
  __int64 v103; // [rsp+70h] [rbp-170h]
  _QWORD *v105; // [rsp+80h] [rbp-160h]
  _QWORD *v106; // [rsp+80h] [rbp-160h]
  __int64 v107; // [rsp+88h] [rbp-158h]
  __int64 v108; // [rsp+88h] [rbp-158h]
  char v109[8]; // [rsp+A0h] [rbp-140h] BYREF
  __int64 v110; // [rsp+A8h] [rbp-138h]
  __int64 v111; // [rsp+B0h] [rbp-130h] BYREF
  int v112; // [rsp+B8h] [rbp-128h]
  __int64 v113; // [rsp+C0h] [rbp-120h] BYREF
  __int64 v114; // [rsp+C8h] [rbp-118h]
  _BYTE *v115; // [rsp+D0h] [rbp-110h] BYREF
  __int64 v116; // [rsp+D8h] [rbp-108h]
  _BYTE v117[64]; // [rsp+E0h] [rbp-100h] BYREF
  _BYTE *v118; // [rsp+120h] [rbp-C0h] BYREF
  __int64 v119; // [rsp+128h] [rbp-B8h]
  _BYTE v120[176]; // [rsp+130h] [rbp-B0h] BYREF

  v9 = a1;
  v11 = *(char **)(a2 + 40);
  v12 = *v11;
  v13 = *((_QWORD *)v11 + 1);
  v109[0] = v12;
  v110 = v13;
  if ( v12 )
  {
    v14 = v12 - 14;
    v92 = word_42E7700[v14];
    switch ( v14 )
    {
      case 0u:
      case 1u:
      case 2u:
      case 3u:
      case 4u:
      case 5u:
      case 6u:
      case 7u:
      case 8u:
      case 9u:
      case 0x2Au:
      case 0x2Bu:
      case 0x2Cu:
      case 0x2Du:
      case 0x2Eu:
      case 0x2Fu:
        v97 = 2;
        break;
      case 0xAu:
      case 0xBu:
      case 0xCu:
      case 0xDu:
      case 0xEu:
      case 0xFu:
      case 0x10u:
      case 0x11u:
      case 0x12u:
      case 0x30u:
      case 0x31u:
      case 0x32u:
      case 0x33u:
      case 0x34u:
      case 0x35u:
        v97 = 3;
        break;
      case 0x13u:
      case 0x14u:
      case 0x15u:
      case 0x16u:
      case 0x17u:
      case 0x18u:
      case 0x19u:
      case 0x1Au:
      case 0x36u:
      case 0x37u:
      case 0x38u:
      case 0x39u:
      case 0x3Au:
      case 0x3Bu:
        v97 = 4;
        break;
      case 0x1Bu:
      case 0x1Cu:
      case 0x1Du:
      case 0x1Eu:
      case 0x1Fu:
      case 0x20u:
      case 0x21u:
      case 0x22u:
      case 0x3Cu:
      case 0x3Du:
      case 0x3Eu:
      case 0x3Fu:
      case 0x40u:
      case 0x41u:
        v97 = 5;
        break;
      case 0x23u:
      case 0x24u:
      case 0x25u:
      case 0x26u:
      case 0x27u:
      case 0x28u:
      case 0x42u:
      case 0x43u:
      case 0x44u:
      case 0x45u:
      case 0x46u:
      case 0x47u:
        v97 = 6;
        break;
      case 0x29u:
        v97 = 7;
        break;
      case 0x48u:
      case 0x49u:
      case 0x4Au:
      case 0x54u:
      case 0x55u:
      case 0x56u:
        v97 = 8;
        break;
      case 0x4Bu:
      case 0x4Cu:
      case 0x4Du:
      case 0x4Eu:
      case 0x4Fu:
      case 0x57u:
      case 0x58u:
      case 0x59u:
      case 0x5Au:
      case 0x5Bu:
        v97 = 9;
        break;
      case 0x50u:
      case 0x51u:
      case 0x52u:
      case 0x53u:
      case 0x5Cu:
      case 0x5Du:
      case 0x5Eu:
      case 0x5Fu:
        v97 = 10;
        break;
    }
    v90 = 0;
  }
  else
  {
    v92 = sub_1F58D30(v109);
    v15 = sub_1F596B0(v109);
    v97 = v15;
    a4 = v15;
    v90 = v16;
  }
  LOBYTE(a4) = v97;
  v17 = *(_QWORD *)(a2 + 72);
  v94 = a4;
  v111 = v17;
  if ( v17 )
    sub_1623A60((__int64)&v111, v17, 2);
  v116 = 0x400000000LL;
  v18 = *(unsigned int *)(a2 + 56);
  v112 = *(_DWORD *)(a2 + 64);
  v19 = v18;
  v118 = v120;
  v119 = 0x800000000LL;
  v20 = v117;
  v115 = v117;
  if ( (unsigned int)v18 > 4 )
  {
    sub_16CD150((__int64)&v115, v117, v18, 16, a5, a6);
    v20 = v115;
  }
  v21 = 16 * v18;
  LODWORD(v116) = v19;
  for ( i = &v20[v21]; i != v20; v20 += 16 )
  {
    if ( v20 )
    {
      *(_QWORD *)v20 = 0;
      *((_DWORD *)v20 + 2) = 0;
    }
  }
  if ( a3 )
  {
    v47 = v92;
    v92 = a3;
    if ( a3 <= v47 )
      v47 = a3;
    v91 = v47;
    if ( !v47 )
    {
LABEL_57:
      v58 = v91;
      v59 = v94;
      do
      {
        LOBYTE(v59) = v97;
        v113 = 0;
        LODWORD(v114) = 0;
        v60 = sub_1D2B300(v9, 0x30u, (__int64)&v113, v59, (__int64)v90, a6);
        v62 = v60;
        a6 = v61;
        if ( v113 )
        {
          v105 = v60;
          v107 = v61;
          sub_161E7C0((__int64)&v113, v113);
          v62 = v105;
          a6 = v107;
        }
        v63 = (unsigned int)v119;
        if ( (unsigned int)v119 >= HIDWORD(v119) )
        {
          v106 = v62;
          v108 = a6;
          sub_16CD150((__int64)&v118, v120, 0, 16, (int)v62, a6);
          v63 = (unsigned int)v119;
          v62 = v106;
          a6 = v108;
        }
        v64 = &v118[16 * v63];
        ++v58;
        *v64 = v62;
        v64[1] = a6;
        LODWORD(v119) = v119 + 1;
      }
      while ( v58 < v92 );
      LODWORD(v94) = v59;
      goto LABEL_64;
    }
  }
  else
  {
    if ( !v92 )
      goto LABEL_64;
    v91 = v92;
  }
  v98 = 0;
  do
  {
    v23 = *(unsigned int *)(a2 + 56);
    if ( !(_DWORD)v23 )
      goto LABEL_50;
    v19 = (unsigned __int64)a1;
    v24 = 0;
    v25 = v103;
    v26 = 0;
    v102 = 40 * v23;
    do
    {
      v28 = (const __m128i *)(v26 + *(_QWORD *)(a2 + 32));
      a7 = _mm_loadu_si128(v28);
      v29 = v28->m128i_i64[0];
      a4 = v28->m128i_u32[2];
      v30 = *(_QWORD *)(v28->m128i_i64[0] + 40) + 16 * a4;
      v31 = *(_BYTE *)v30;
      v32 = *(_QWORD *)(v30 + 8);
      LOBYTE(v113) = v31;
      v114 = v32;
      if ( v31 )
      {
        if ( (unsigned __int8)(v31 - 14) > 0x5Fu )
          goto LABEL_19;
        switch ( v31 )
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
            LOBYTE(v34) = 3;
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
            LOBYTE(v34) = 4;
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
            LOBYTE(v34) = 5;
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
            LOBYTE(v34) = 6;
            break;
          case 55:
            LOBYTE(v34) = 7;
            break;
          case 86:
          case 87:
          case 88:
          case 98:
          case 99:
          case 100:
            LOBYTE(v34) = 8;
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
            LOBYTE(v34) = 9;
            break;
          case 94:
          case 95:
          case 96:
          case 97:
          case 106:
          case 107:
          case 108:
          case 109:
            LOBYTE(v34) = 10;
            break;
          default:
            LOBYTE(v34) = 2;
            break;
        }
        v36 = 0;
      }
      else
      {
        v99 = a4;
        v33 = sub_1F58D20(&v113);
        a4 = v99;
        if ( !v33 )
        {
LABEL_19:
          v27 = &v115[v24];
          *(_QWORD *)v27 = v29;
          *((_DWORD *)v27 + 2) = a4;
          goto LABEL_20;
        }
        v34 = sub_1F596B0(&v113);
        v25 = v34;
        v36 = v35;
      }
      LOBYTE(v25) = v34;
      v100 = a1[2];
      v101 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v100 + 48LL);
      v37 = sub_1E0A0C0(a1[4]);
      if ( v101 == sub_1D13A20 )
      {
        v38 = 8 * sub_15A9520(v37, 0);
        if ( v38 == 32 )
        {
          v39 = 5;
        }
        else if ( v38 > 0x20 )
        {
          v39 = 6;
          if ( v38 != 64 )
          {
            v39 = 0;
            if ( v38 == 128 )
              v39 = 7;
          }
        }
        else
        {
          v39 = 3;
          if ( v38 != 8 )
            v39 = 4 * (v38 == 16);
        }
      }
      else
      {
        v39 = v101(v100, v37);
      }
      v40 = v96;
      LOBYTE(v40) = v39;
      *(_QWORD *)&v41 = sub_1D38BB0((__int64)a1, v98, (__int64)&v111, v40, 0, 0, a7, a8, a9, 0);
      v42 = sub_1D332F0(
              a1,
              106,
              (__int64)&v111,
              (unsigned int)v25,
              v36,
              0,
              *(double *)a7.m128i_i64,
              a8,
              a9,
              a7.m128i_i64[0],
              a7.m128i_u64[1],
              v41);
      v44 = v43;
      v45 = v42;
      v46 = (unsigned __int64)v115;
      *(_QWORD *)&v115[v24] = v45;
      *(_DWORD *)(v46 + v24 + 8) = v44;
LABEL_20:
      v26 += 40;
      v24 += 16;
    }
    while ( v26 != v102 );
    v103 = v25;
LABEL_50:
    v48 = *(unsigned __int16 *)(a2 + 24);
    if ( (_WORD)v48 == 135 )
    {
      v74 = v94;
      LOBYTE(v74) = v97;
      *((_QWORD *)&v87 + 1) = (unsigned int)v116;
      *(_QWORD *)&v87 = v115;
      v94 = v74;
      v72 = sub_1D359D0(a1, 134, (__int64)&v111, (unsigned int)v74, v90, 0, *(double *)a7.m128i_i64, a8, a9, v87);
      goto LABEL_75;
    }
    if ( (__int16)v48 > 135 )
    {
      if ( (v48 & 0xFFF7) != 0x94 )
        goto LABEL_74;
      v75 = *((_QWORD *)v115 + 2);
      v76 = *(_BYTE *)(v75 + 88);
      v77 = *(_QWORD *)(v75 + 96);
      LOBYTE(v113) = v76;
      v114 = v77;
      if ( v76 )
      {
        switch ( v76 )
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
            LOBYTE(v78) = 2;
            v79 = 0;
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
            LOBYTE(v78) = 3;
            v79 = 0;
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
            LOBYTE(v78) = 4;
            v79 = 0;
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
            LOBYTE(v78) = 5;
            v79 = 0;
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
            LOBYTE(v78) = 6;
            v79 = 0;
            break;
          case 55:
            LOBYTE(v78) = 7;
            v79 = 0;
            break;
          case 86:
          case 87:
          case 88:
          case 98:
          case 99:
          case 100:
            LOBYTE(v78) = 8;
            v79 = 0;
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
            LOBYTE(v78) = 9;
            v79 = 0;
            break;
          case 94:
          case 95:
          case 96:
          case 97:
          case 106:
          case 107:
          case 108:
          case 109:
            LOBYTE(v78) = 10;
            v79 = 0;
            break;
          default:
            ++*(_DWORD *)(v19 + 888);
            BUG();
        }
      }
      else
      {
        v78 = sub_1F596B0(&v113);
        v89 = v78;
      }
      v80 = v89;
      LOBYTE(v80) = v78;
      v89 = v80;
      *(_QWORD *)&v81 = sub_1D2EF30(a1, (unsigned int)v80, v79, a4, a5, a6);
      v82 = v94;
      LOBYTE(v82) = v97;
      v94 = v82;
      v53 = sub_1D332F0(
              a1,
              *(unsigned __int16 *)(a2 + 24),
              (__int64)&v111,
              v82,
              v90,
              0,
              *(double *)a7.m128i_i64,
              a8,
              a9,
              *(_QWORD *)v115,
              *((_QWORD *)v115 + 1),
              v81);
LABEL_54:
      v55 = v54;
      v19 = (unsigned __int64)v53;
      v56 = (unsigned int)v119;
      if ( (unsigned int)v119 >= HIDWORD(v119) )
        goto LABEL_76;
    }
    else
    {
      if ( (unsigned __int16)(v48 - 122) <= 4u )
      {
        v49 = v88;
        v50 = *(_QWORD *)(*(_QWORD *)v115 + 40LL) + 16LL * *((unsigned int *)v115 + 2);
        LOBYTE(v49) = *(_BYTE *)v50;
        v88 = v49;
        *(_QWORD *)&v51 = sub_1D324C0(
                            a1,
                            v49,
                            *(_QWORD *)(v50 + 8),
                            *((_QWORD *)v115 + 2),
                            *((_QWORD *)v115 + 3),
                            *(double *)a7.m128i_i64,
                            a8,
                            *(double *)a9.m128i_i64);
        v52 = v94;
        LOBYTE(v52) = v97;
        v94 = v52;
        v53 = sub_1D332F0(
                a1,
                *(unsigned __int16 *)(a2 + 24),
                (__int64)&v111,
                v52,
                v90,
                0,
                *(double *)a7.m128i_i64,
                a8,
                a9,
                *(_QWORD *)v115,
                *((_QWORD *)v115 + 1),
                v51);
        goto LABEL_54;
      }
LABEL_74:
      v71 = v94;
      LOBYTE(v71) = v97;
      v94 = v71;
      *((_QWORD *)&v85 + 1) = (unsigned int)v116;
      *(_QWORD *)&v85 = v115;
      v72 = sub_1D359D0(
              a1,
              v48,
              (__int64)&v111,
              (unsigned int)v71,
              v90,
              *(unsigned __int16 *)(a2 + 80),
              *(double *)a7.m128i_i64,
              a8,
              a9,
              v85);
LABEL_75:
      a4 = v86;
      v55 = v73;
      v19 = (unsigned __int64)v72;
      v56 = (unsigned int)v119;
      if ( (unsigned int)v119 >= HIDWORD(v119) )
      {
LABEL_76:
        sub_16CD150((__int64)&v118, v120, 0, 16, a5, a6);
        v56 = (unsigned int)v119;
      }
    }
    v57 = (unsigned __int64 *)&v118[16 * v56];
    *v57 = v19;
    v57[1] = v55;
    ++v98;
    LODWORD(v119) = v119 + 1;
  }
  while ( v98 != v91 );
  v9 = a1;
  if ( v91 < v92 )
    goto LABEL_57;
LABEL_64:
  v65 = v94;
  LOBYTE(v65) = v97;
  v66 = v9[6];
  v95 = v65;
  v67 = sub_1D15020(v97, v92);
  v68 = 0;
  if ( !v67 )
  {
    v67 = sub_1F593D0(v66, v95, v90, v92);
    v68 = v83;
  }
  *((_QWORD *)&v84 + 1) = (unsigned int)v119;
  *(_QWORD *)&v84 = v118;
  v69 = sub_1D359D0(v9, 104, (__int64)&v111, v67, v68, 0, *(double *)a7.m128i_i64, a8, a9, v84);
  if ( v115 != v117 )
    _libc_free((unsigned __int64)v115);
  if ( v118 != v120 )
    _libc_free((unsigned __int64)v118);
  if ( v111 )
    sub_161E7C0((__int64)&v111, v111);
  return v69;
}
