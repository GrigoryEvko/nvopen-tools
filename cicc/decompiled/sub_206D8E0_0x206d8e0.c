// Function: sub_206D8E0
// Address: 0x206d8e0
//
void __fastcall sub_206D8E0(__int64 a1, __int64 a2, __m128i a3, __m128i a4, __m128i a5)
{
  unsigned int v5; // r14d
  __int64 v7; // rbx
  __int64 v8; // rdi
  __int64 v9; // rax
  int v10; // r8d
  int v11; // r9d
  _BYTE *v12; // rax
  _BYTE *v13; // rdx
  __int64 **v14; // rax
  __int64 *v15; // rbx
  __int64 v16; // rdx
  __int64 v17; // r13
  __int64 v18; // rax
  unsigned int v19; // edx
  __int64 v20; // rax
  __int64 *v21; // rax
  __int64 v22; // r13
  char v23; // al
  int v24; // edx
  __int64 v25; // rdx
  __m128i v26; // xmm0
  __int64 v27; // rax
  __int64 v28; // rbx
  __int64 v29; // r13
  __int64 v30; // rax
  char v31; // bl
  int v32; // edx
  unsigned __int8 v33; // al
  __int64 v34; // rdx
  __int64 v35; // rdx
  __int64 v36; // rbx
  unsigned __int64 v37; // rdx
  unsigned int v38; // ebx
  __int64 v39; // r13
  unsigned int v40; // edx
  unsigned __int64 v41; // r8
  __int64 *v42; // r9
  __int64 v43; // rax
  __int64 **v44; // rax
  __int64 v45; // rax
  __int64 *v46; // r9
  __int64 **v47; // rax
  __int32 v48; // edx
  __int64 v49; // r10
  __int64 v50; // r11
  __int64 v51; // rax
  const void **v52; // r8
  __m128i *v53; // rax
  __int64 v54; // rsi
  __int64 *v55; // rax
  int v56; // edx
  int v57; // edi
  __int64 *v58; // rdx
  unsigned __int64 v59; // rax
  const void *v60; // r10
  int v61; // r8d
  __int64 v62; // r9
  _QWORD *v63; // rdi
  __int64 *v64; // rbx
  unsigned __int64 v65; // r13
  __int64 v66; // r14
  __int64 v67; // rax
  int v68; // edx
  __int64 v69; // r9
  const void ***v70; // rcx
  int v71; // r8d
  __m128i *v72; // rax
  __int64 v73; // rsi
  __int64 *v74; // rbx
  int v75; // edx
  int v76; // r13d
  __int64 *v77; // rax
  unsigned int v78; // edx
  __int64 *v79; // rax
  int v80; // edx
  char v81; // al
  char v82; // al
  __int128 v83; // [rsp-10h] [rbp-1F0h]
  __int128 v84; // [rsp-10h] [rbp-1F0h]
  size_t n; // [rsp+10h] [rbp-1D0h]
  size_t na; // [rsp+10h] [rbp-1D0h]
  __int64 v87; // [rsp+18h] [rbp-1C8h]
  const void **srcc; // [rsp+28h] [rbp-1B8h]
  void *srcd; // [rsp+28h] [rbp-1B8h]
  void *srcb; // [rsp+28h] [rbp-1B8h]
  unsigned __int8 src; // [rsp+28h] [rbp-1B8h]
  unsigned __int8 srca; // [rsp+28h] [rbp-1B8h]
  unsigned int v94; // [rsp+40h] [rbp-1A0h]
  int v95; // [rsp+40h] [rbp-1A0h]
  int v96; // [rsp+48h] [rbp-198h]
  int v97; // [rsp+48h] [rbp-198h]
  unsigned int v98; // [rsp+4Ch] [rbp-194h]
  __int64 *v99; // [rsp+50h] [rbp-190h]
  unsigned int v100; // [rsp+58h] [rbp-188h]
  __int64 *v101; // [rsp+58h] [rbp-188h]
  int v102; // [rsp+58h] [rbp-188h]
  int v103; // [rsp+58h] [rbp-188h]
  __int64 *v104; // [rsp+60h] [rbp-180h]
  int v105; // [rsp+60h] [rbp-180h]
  int v106; // [rsp+68h] [rbp-178h]
  const void ***v107; // [rsp+68h] [rbp-178h]
  __int64 *v108; // [rsp+90h] [rbp-150h] BYREF
  __int64 *v109; // [rsp+98h] [rbp-148h] BYREF
  __m128i v110; // [rsp+A0h] [rbp-140h] BYREF
  _QWORD *v111; // [rsp+B0h] [rbp-130h]
  __int64 v112; // [rsp+B8h] [rbp-128h]
  _QWORD v113[2]; // [rsp+C0h] [rbp-120h] BYREF
  _QWORD *v114; // [rsp+D0h] [rbp-110h] BYREF
  __int64 v115; // [rsp+D8h] [rbp-108h]
  _QWORD v116[6]; // [rsp+E0h] [rbp-100h] BYREF
  __m128i *v117; // [rsp+110h] [rbp-D0h] BYREF
  __int64 v118; // [rsp+118h] [rbp-C8h]
  _BYTE v119[64]; // [rsp+120h] [rbp-C0h] BYREF
  _BYTE *v120; // [rsp+160h] [rbp-80h] BYREF
  __int64 v121; // [rsp+168h] [rbp-78h]
  _BYTE v122[112]; // [rsp+170h] [rbp-70h] BYREF

  v7 = *(_QWORD *)a2;
  v117 = (__m128i *)v119;
  v8 = *(_QWORD *)(*(_QWORD *)(a1 + 552) + 32LL);
  v118 = 0x400000000LL;
  v9 = sub_1E0A0C0(v8);
  sub_20C7CE0(*(_QWORD *)(*(_QWORD *)(a1 + 552) + 16LL), v9, v7, &v117, 0, 0);
  v106 = v118;
  if ( !(_DWORD)v118 )
    goto LABEL_2;
  v12 = v122;
  v121 = 0x400000000LL;
  v120 = v122;
  if ( (unsigned int)v118 > 4 )
  {
    sub_16CD150((__int64)&v120, v122, (unsigned int)v118, 16, v10, v11);
    v12 = v120;
  }
  v13 = &v12[16 * v106];
  LODWORD(v121) = v106;
  do
  {
    if ( v12 )
    {
      *(_QWORD *)v12 = 0;
      *((_DWORD *)v12 + 2) = 0;
    }
    v12 += 16;
  }
  while ( v13 != v12 );
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v14 = *(__int64 ***)(a2 - 8);
  else
    v14 = (__int64 **)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  v15 = sub_20685E0(a1, *v14, a3, a4, a5);
  v17 = v16;
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v18 = *(_QWORD *)(a2 - 8);
  else
    v18 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
  v104 = sub_20685E0(a1, *(__int64 **)(v18 + 24), a3, a4, a5);
  v100 = v19;
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v20 = *(_QWORD *)(a2 - 8);
  else
    v20 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
  v21 = sub_20685E0(a1, *(__int64 **)(v20 + 48), a3, a4, a5);
  v113[1] = v17;
  v99 = v21;
  v22 = v15[5] + 16LL * (unsigned int)v17;
  v111 = v113;
  v112 = 0x100000001LL;
  v113[0] = v15;
  v23 = *(_BYTE *)v22;
  v96 = v24;
  v25 = *(_QWORD *)(v22 + 8);
  LOBYTE(v114) = v23;
  v115 = v25;
  if ( v23 )
  {
    if ( (unsigned __int8)(v23 - 14) > 0x5Fu )
      v98 = 134;
    else
      v98 = 135;
  }
  else
  {
    v98 = 134 - (!sub_1F58D20((__int64)&v114) - 1);
  }
  v26 = _mm_loadu_si128(v117);
  v27 = *(_QWORD *)(a1 + 552);
  v110 = v26;
  v28 = *(_QWORD *)(v27 + 48);
  v29 = *(_QWORD *)(v27 + 16);
  while ( 1 )
  {
    sub_1F40D10((__int64)&v114, v29, v28, v110.m128i_i64[0], v110.m128i_i64[1]);
    if ( !(_BYTE)v114 )
      break;
    sub_1F40D10((__int64)&v114, v29, v28, v110.m128i_i64[0], v110.m128i_i64[1]);
    v30 = v110.m128i_u8[0];
    if ( (_BYTE)v115 == v110.m128i_i8[0] )
    {
      if ( (_BYTE)v115 )
        goto LABEL_26;
      if ( v116[0] == v110.m128i_i64[1] )
        goto LABEL_80;
    }
    sub_1F40D10((__int64)&v114, v29, v28, v110.m128i_i64[0], v110.m128i_i64[1]);
    v110.m128i_i8[0] = v115;
    v110.m128i_i64[1] = v116[0];
  }
  v30 = v110.m128i_u8[0];
  if ( v110.m128i_i8[0] )
  {
LABEL_26:
    if ( (unsigned __int8)(v30 - 14) <= 0x5Fu )
    {
      if ( *(_QWORD *)(v29 + 8 * v30 + 120) && (*(_BYTE *)(v29 + 259LL * (unsigned __int8)v30 + 2557) & 0xFB) == 0 )
      {
        v31 = 0;
        goto LABEL_36;
      }
      goto LABEL_81;
    }
LABEL_35:
    v31 = 0;
    goto LABEL_36;
  }
LABEL_80:
  if ( !sub_1F58D20((__int64)&v110) )
    goto LABEL_35;
LABEL_81:
  v31 = 1;
LABEL_36:
  v114 = (_QWORD *)sub_14B2890(a2, (__int64 *)&v108, (__int64 *)&v109, 0, 0);
  LODWORD(v115) = v32;
  switch ( (int)v114 )
  {
    case 1:
      v94 = 114;
      goto LABEL_38;
    case 2:
      v94 = 116;
      goto LABEL_38;
    case 3:
      v94 = 115;
      goto LABEL_38;
    case 4:
      v94 = 117;
      goto LABEL_38;
    case 5:
      if ( HIDWORD(v114) == 2 )
        goto LABEL_109;
      if ( HIDWORD(v114) != 3 )
      {
        if ( HIDWORD(v114) != 1 )
          goto LABEL_48;
        goto LABEL_90;
      }
      src = v110.m128i_i8[0];
      if ( sub_204D480(v29, 0xB4u, v110.m128i_u8[0]) )
        goto LABEL_109;
      if ( sub_204D480(v29, 0xB6u, src) )
        goto LABEL_90;
      if ( !v31 )
        goto LABEL_48;
      v81 = sub_1D15870(v110.m128i_i8);
      if ( !sub_204D480(v29, 0xB4u, v81) )
      {
LABEL_90:
        v94 = 182;
        goto LABEL_38;
      }
LABEL_109:
      v94 = 180;
      goto LABEL_38;
    case 6:
      if ( HIDWORD(v114) == 2 )
        goto LABEL_114;
      if ( HIDWORD(v114) != 3 )
      {
        if ( HIDWORD(v114) != 1 )
          goto LABEL_48;
        goto LABEL_86;
      }
      srca = v110.m128i_i8[0];
      if ( sub_204D480(v29, 0xB5u, v110.m128i_u8[0]) )
        goto LABEL_114;
      if ( sub_204D480(v29, 0xB7u, srca) )
        goto LABEL_86;
      if ( !v31 )
        goto LABEL_48;
      v82 = sub_1D15870(v110.m128i_i8);
      if ( !sub_204D480(v29, 0xB5u, v82) )
      {
LABEL_86:
        v94 = 183;
        goto LABEL_38;
      }
LABEL_114:
      v94 = 181;
LABEL_38:
      v33 = v110.m128i_i8[0];
      v34 = 1;
      if ( v110.m128i_i8[0] == 1 )
        goto LABEL_39;
      if ( !v110.m128i_i8[0] )
      {
        if ( !v31 || !sub_1F58D20((__int64)&v110) )
          goto LABEL_48;
        v33 = sub_1F596B0((__int64)&v110);
        goto LABEL_42;
      }
      v34 = v110.m128i_u8[0];
      if ( *(_QWORD *)(v29 + 8LL * v110.m128i_u8[0] + 120) )
      {
LABEL_39:
        if ( (*(_BYTE *)(v94 + v29 + 259 * v34 + 2422) & 0xFB) == 0 )
        {
LABEL_44:
          v36 = *(_QWORD *)(*(_QWORD *)(a2 - 72) + 8LL);
          if ( !v36 )
          {
LABEL_100:
            v104 = sub_20685E0(a1, v108, v26, a4, a5);
            v100 = v78;
            v79 = sub_20685E0(a1, v109, v26, a4, a5);
            LODWORD(v112) = 0;
            v99 = v79;
            v96 = v80;
            v37 = 0;
            v98 = v94;
            goto LABEL_49;
          }
          while ( *((_BYTE *)sub_1648700(v36) + 16) == 79 )
          {
            v36 = *(_QWORD *)(v36 + 8);
            if ( !v36 )
              goto LABEL_100;
          }
LABEL_48:
          v37 = (unsigned int)v112;
          goto LABEL_49;
        }
        if ( !v31 )
          goto LABEL_48;
LABEL_41:
        if ( (unsigned __int8)(v110.m128i_i8[0] - 14) <= 0x5Fu )
        {
          switch ( v110.m128i_i8[0] )
          {
            case 0x18:
            case 0x19:
            case 0x1A:
            case 0x1B:
            case 0x1C:
            case 0x1D:
            case 0x1E:
            case 0x1F:
            case 0x20:
            case 0x3E:
            case 0x3F:
            case 0x40:
            case 0x41:
            case 0x42:
            case 0x43:
              v33 = 3;
              break;
            case 0x21:
            case 0x22:
            case 0x23:
            case 0x24:
            case 0x25:
            case 0x26:
            case 0x27:
            case 0x28:
            case 0x44:
            case 0x45:
            case 0x46:
            case 0x47:
            case 0x48:
            case 0x49:
              v33 = 4;
              break;
            case 0x29:
            case 0x2A:
            case 0x2B:
            case 0x2C:
            case 0x2D:
            case 0x2E:
            case 0x2F:
            case 0x30:
            case 0x4A:
            case 0x4B:
            case 0x4C:
            case 0x4D:
            case 0x4E:
            case 0x4F:
              v33 = 5;
              break;
            case 0x31:
            case 0x32:
            case 0x33:
            case 0x34:
            case 0x35:
            case 0x36:
            case 0x50:
            case 0x51:
            case 0x52:
            case 0x53:
            case 0x54:
            case 0x55:
              v33 = 6;
              break;
            case 0x37:
              v33 = 7;
              break;
            case 0x56:
            case 0x57:
            case 0x58:
            case 0x62:
            case 0x63:
            case 0x64:
              v33 = 8;
              break;
            case 0x59:
            case 0x5A:
            case 0x5B:
            case 0x5C:
            case 0x5D:
            case 0x65:
            case 0x66:
            case 0x67:
            case 0x68:
            case 0x69:
              v33 = 9;
              break;
            case 0x5E:
            case 0x5F:
            case 0x60:
            case 0x61:
            case 0x6A:
            case 0x6B:
            case 0x6C:
            case 0x6D:
              v33 = 10;
              break;
            default:
              v33 = 2;
              break;
          }
          goto LABEL_103;
        }
LABEL_42:
        v35 = 1;
        if ( v33 == 1 )
          goto LABEL_43;
        if ( !v33 )
          goto LABEL_48;
LABEL_103:
        v35 = v33;
        if ( !*(_QWORD *)(v29 + 8LL * v33 + 120) )
          goto LABEL_48;
LABEL_43:
        if ( (*(_BYTE *)(v94 + 259 * v35 + v29 + 2422) & 0xFB) != 0 )
          goto LABEL_48;
        goto LABEL_44;
      }
      v37 = (unsigned int)v112;
      if ( v31 )
        goto LABEL_41;
LABEL_49:
      v38 = v100;
      v39 = 0;
      v95 = v100 + v106;
      v97 = v96 - v100;
      while ( 1 )
      {
        v60 = v111;
        v61 = v37;
        v62 = 16 * v37;
        v114 = v116;
        v115 = 0x300000000LL;
        if ( v37 <= 3 )
        {
          if ( !v62 )
            goto LABEL_51;
          v63 = v116;
        }
        else
        {
          na = 16 * v37;
          srcd = v111;
          v102 = v37;
          sub_16CD150((__int64)&v114, v116, v37, 16, v37, v62);
          v61 = v102;
          v60 = srcd;
          v62 = na;
          v63 = &v114[2 * (unsigned int)v115];
        }
        v103 = v61;
        memcpy(v63, v60, v62);
        LODWORD(v62) = v115;
        v61 = v103;
LABEL_51:
        v40 = v61 + v62;
        v41 = v38;
        v42 = (__int64 *)v38;
        LODWORD(v115) = v40;
        v43 = v40;
        if ( HIDWORD(v115) <= v40 )
        {
          sub_16CD150((__int64)&v114, v116, 0, 16, v38, v38);
          v43 = (unsigned int)v115;
          v41 = v38;
          v42 = (__int64 *)v38;
        }
        v44 = (__int64 **)&v114[2 * v43];
        v44[1] = v42;
        *v44 = v104;
        v45 = (unsigned int)(v115 + 1);
        LODWORD(v115) = v45;
        v46 = (__int64 *)(v97 + v38);
        if ( HIDWORD(v115) <= (unsigned int)v45 )
        {
          srcb = (void *)v41;
          sub_16CD150((__int64)&v114, v116, 0, 16, v41, (int)v46);
          v45 = (unsigned int)v115;
          v41 = (unsigned __int64)srcb;
          v46 = (__int64 *)(v97 + v38);
        }
        v47 = (__int64 **)&v114[2 * v45];
        v47[1] = v46;
        *v47 = v99;
        v48 = *(_DWORD *)(a1 + 536);
        v101 = *(__int64 **)(a1 + 552);
        v49 = (__int64)v114;
        LODWORD(v115) = v115 + 1;
        v50 = (unsigned int)v115;
        v51 = 16 * v41 + v104[5];
        v52 = *(const void ***)(v51 + 8);
        LOBYTE(v5) = *(_BYTE *)v51;
        v110.m128i_i32[2] = v48;
        v110.m128i_i64[0] = 0;
        v53 = *(__m128i **)a1;
        if ( *(_QWORD *)a1 )
        {
          if ( &v110 != &v53[3] )
          {
            v54 = v53[3].m128i_i64[0];
            v110.m128i_i64[0] = v54;
            if ( v54 )
            {
              n = (size_t)v114;
              v87 = (unsigned int)v115;
              srcc = v52;
              sub_1623A60((__int64)&v110, v54, 2);
              v49 = n;
              v50 = v87;
              v52 = srcc;
            }
          }
        }
        *((_QWORD *)&v83 + 1) = v50;
        *(_QWORD *)&v83 = v49;
        v55 = sub_1D359D0(
                v101,
                v98,
                (__int64)&v110,
                v5,
                v52,
                0,
                *(double *)v26.m128i_i64,
                *(double *)a4.m128i_i64,
                a5,
                v83);
        v57 = v56;
        v58 = v55;
        v59 = (unsigned __int64)v120;
        *(_QWORD *)&v120[v39] = v58;
        *(_DWORD *)(v59 + v39 + 8) = v57;
        if ( v110.m128i_i64[0] )
          sub_161E7C0((__int64)&v110, v110.m128i_i64[0]);
        if ( v114 != v116 )
          _libc_free((unsigned __int64)v114);
        v39 += 16;
        if ( v95 == ++v38 )
          break;
        v37 = (unsigned int)v112;
      }
      v64 = *(__int64 **)(a1 + 552);
      v65 = (unsigned __int64)v120;
      v66 = (unsigned int)v121;
      v67 = sub_1D25C30((__int64)v64, (unsigned __int8 *)v117, (unsigned int)v118);
      v114 = 0;
      v70 = (const void ***)v67;
      v71 = v68;
      v72 = *(__m128i **)a1;
      LODWORD(v115) = *(_DWORD *)(a1 + 536);
      if ( v72 )
      {
        if ( &v114 != (_QWORD **)&v72[3] )
        {
          v73 = v72[3].m128i_i64[0];
          v114 = (_QWORD *)v73;
          if ( v73 )
          {
            v105 = v68;
            v107 = v70;
            sub_1623A60((__int64)&v114, v73, 2);
            v71 = v105;
            v70 = v107;
          }
        }
      }
      *((_QWORD *)&v84 + 1) = v66;
      *(_QWORD *)&v84 = v65;
      v74 = sub_1D36D80(
              v64,
              51,
              (__int64)&v114,
              v70,
              v71,
              *(double *)v26.m128i_i64,
              *(double *)a4.m128i_i64,
              a5,
              v69,
              v84);
      v76 = v75;
      v110.m128i_i64[0] = a2;
      v77 = sub_205F5C0(a1 + 8, v110.m128i_i64);
      v77[1] = (__int64)v74;
      *((_DWORD *)v77 + 4) = v76;
      if ( v114 )
        sub_161E7C0((__int64)&v114, (__int64)v114);
      if ( v111 != v113 )
        _libc_free((unsigned __int64)v111);
      if ( v120 != v122 )
        _libc_free((unsigned __int64)v120);
LABEL_2:
      if ( v117 != (__m128i *)v119 )
        _libc_free((unsigned __int64)v117);
      return;
    default:
      goto LABEL_48;
  }
}
