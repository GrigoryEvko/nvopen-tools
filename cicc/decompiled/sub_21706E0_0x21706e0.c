// Function: sub_21706E0
// Address: 0x21706e0
//
void __fastcall sub_21706E0(__int64 a1, __int64 *a2, __int64 a3, __m128i a4, double a5, __m128i a6)
{
  __int64 v6; // r12
  __int64 v8; // rsi
  __int64 v9; // rax
  unsigned __int64 v10; // rsi
  unsigned __int64 v11; // rsi
  unsigned int v12; // edx
  __int64 v13; // rax
  __int64 v14; // rdx
  char *v15; // rsi
  __int64 v16; // rcx
  __int64 v17; // rdi
  unsigned __int8 v18; // al
  char v19; // di
  __int64 v20; // rcx
  _BYTE *v21; // rcx
  __int64 *v22; // r8
  __int64 *v23; // r9
  __int64 v24; // rax
  __int64 *v25; // rax
  unsigned __int64 v26; // r13
  __int64 v27; // rax
  char v28; // cl
  __int64 v29; // rax
  __int64 v30; // rax
  const void **v31; // rdx
  unsigned int v32; // eax
  __int64 v33; // r15
  __int64 **v34; // rax
  __int128 v35; // rax
  __int64 *v36; // rdx
  __int64 v37; // rax
  char v38; // al
  __int64 v39; // rsi
  char *v40; // rdx
  const void **v41; // rdx
  int v42; // r13d
  unsigned __int8 v43; // cl
  __int64 v44; // rdx
  __int64 v45; // r15
  __int64 v46; // rdx
  __int64 v47; // rax
  _BYTE *v48; // rcx
  int v49; // r12d
  __int64 *v50; // r14
  __int64 v51; // rbx
  _QWORD *v52; // rdx
  unsigned __int8 *v53; // rsi
  __int64 v54; // rax
  int v55; // edx
  __int64 v56; // r9
  __int64 v57; // rax
  int v58; // r9d
  __int64 v59; // r8
  __int64 *v60; // r14
  __int64 v61; // rax
  unsigned int v62; // edx
  unsigned int v63; // r15d
  int v64; // r12d
  __int64 v65; // rbx
  __int64 v66; // r13
  __int64 *v67; // rax
  _BYTE *v68; // rdx
  __int64 *v69; // rdx
  __int64 *v70; // r8
  __int64 *v71; // r9
  __int64 v72; // rax
  __int64 **v73; // rax
  _BYTE *v74; // rdi
  const void **v75; // rdx
  unsigned __int8 v76; // al
  int v77; // [rsp-10h] [rbp-400h]
  __int128 v78; // [rsp-10h] [rbp-400h]
  int v79; // [rsp-8h] [rbp-3F8h]
  __int64 *v81; // [rsp+10h] [rbp-3E0h]
  __int64 *v82; // [rsp+18h] [rbp-3D8h]
  __int16 v83; // [rsp+24h] [rbp-3CCh]
  int v84; // [rsp+3Ch] [rbp-3B4h]
  const void **v86; // [rsp+48h] [rbp-3A8h]
  __int64 v87; // [rsp+50h] [rbp-3A0h]
  __int64 *v88; // [rsp+50h] [rbp-3A0h]
  __int64 *v89; // [rsp+50h] [rbp-3A0h]
  __int64 v90; // [rsp+58h] [rbp-398h]
  __int64 *v91; // [rsp+58h] [rbp-398h]
  __int64 v92; // [rsp+60h] [rbp-390h]
  unsigned int v93; // [rsp+68h] [rbp-388h]
  __int64 v94; // [rsp+70h] [rbp-380h] BYREF
  int v95; // [rsp+78h] [rbp-378h]
  __int64 v96; // [rsp+80h] [rbp-370h] BYREF
  const void **v97; // [rsp+88h] [rbp-368h]
  _BYTE *v98; // [rsp+90h] [rbp-360h]
  __int64 v99; // [rsp+98h] [rbp-358h]
  _BYTE v100[32]; // [rsp+A0h] [rbp-350h] BYREF
  _BYTE *v101; // [rsp+C0h] [rbp-330h] BYREF
  __int64 v102; // [rsp+C8h] [rbp-328h]
  _BYTE v103[80]; // [rsp+D0h] [rbp-320h] BYREF
  _BYTE *v104; // [rsp+120h] [rbp-2D0h] BYREF
  __int64 v105; // [rsp+128h] [rbp-2C8h]
  _BYTE v106[128]; // [rsp+130h] [rbp-2C0h] BYREF
  __int64 *v107; // [rsp+1B0h] [rbp-240h] BYREF
  __int64 v108; // [rsp+1B8h] [rbp-238h]
  _QWORD v109[70]; // [rsp+1C0h] [rbp-230h] BYREF

  v8 = *(_QWORD *)(a1 + 72);
  v94 = v8;
  if ( v8 )
    sub_1623A60((__int64)&v94, v8, 2);
  v95 = *(_DWORD *)(a1 + 64);
  v9 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 32) + 40LL) + 88LL);
  v10 = *(_QWORD *)(v9 + 24);
  if ( *(_DWORD *)(v9 + 32) > 0x40u )
    v10 = *(_QWORD *)v10;
  if ( (v10 & 0xF000000) != 0 )
    v11 = ((unsigned __int64)(BYTE3(v10) & 0xF0 | (BYTE3(v10) + 3) & 3 | (4 * (unsigned __int8)(v10 >> 26) + 12) & 0xCu) << 24)
        | v10 & 0xFFFFFFFF00FFFFFFLL;
  else
    v11 = ((unsigned __int64)(((unsigned __int8)(v10 >> 26) + 1) & 3) << 26) | v10 & 0xFFFFFFFFF3FFFFFFLL;
  v109[0] = sub_1D38BB0((__int64)a2, v11, (__int64)&v94, 6, 0, 1, a4, a5, a6, 0);
  v13 = v12;
  v14 = *(_QWORD *)(a1 + 32);
  v107 = v109;
  v15 = *(char **)(a1 + 40);
  v16 = *(unsigned int *)(v14 + 88);
  v17 = *(_QWORD *)(v14 + 80);
  v109[1] = v13;
  v108 = 0x2000000001LL;
  v22 = *(__int64 **)(v14 + 120);
  v18 = *v15;
  v19 = *(_BYTE *)(*(_QWORD *)(v17 + 40) + 16 * v16);
  v20 = *(unsigned int *)(v14 + 128);
  v100[0] = v18;
  v21 = (_BYTE *)(v22[5] + 16 * v20);
  v100[1] = v19;
  LOBYTE(v21) = *v21;
  v98 = v100;
  LODWORD(v22) = v77;
  LODWORD(v23) = v79;
  v100[2] = (_BYTE)v21;
  v99 = 0x2000000003LL;
  if ( v18 == 91 )
  {
    v83 = 3176;
  }
  else if ( v18 <= 0x5Bu )
  {
    v83 = 3175;
    if ( v18 != 42 )
    {
      v83 = 3180;
      if ( v19 != 5 )
        v83 = (v19 != 42) + 3181;
    }
  }
  else if ( v18 == 92 )
  {
    v83 = 3177;
    if ( v19 != 42 )
      v83 = (v19 != 43) + 3178;
  }
  else
  {
    v83 = 3174;
  }
  v84 = *(_DWORD *)(a1 + 56);
  if ( v84 == 2 )
  {
    v75 = (const void **)*((_QWORD *)v15 + 1);
    LOBYTE(v96) = v18;
    v97 = v75;
    goto LABEL_64;
  }
  v93 = 2;
  while ( 1 )
  {
    v26 = *(_QWORD *)(v14 + 40LL * v93 + 8);
    v6 = *(_QWORD *)(v14 + 40LL * v93);
    v27 = *(_QWORD *)(v6 + 40) + 16LL * (unsigned int)v26;
    v28 = *(_BYTE *)v27;
    v29 = *(_QWORD *)(v27 + 8);
    LOBYTE(v104) = v28;
    v105 = v29;
    if ( v28 )
    {
      if ( (unsigned __int8)(v28 - 14) <= 0x5Fu )
      {
        switch ( v28 )
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
            v38 = 3;
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
            v38 = 4;
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
            v38 = 5;
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
            v38 = 6;
            break;
          case 55:
            v38 = 7;
            break;
          case 86:
          case 87:
          case 88:
          case 98:
          case 99:
          case 100:
            v38 = 8;
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
            v38 = 9;
            break;
          case 94:
          case 95:
          case 96:
          case 97:
          case 106:
          case 107:
          case 108:
          case 109:
            v38 = 10;
            break;
          default:
            v38 = 2;
            break;
        }
        v39 = v87;
        v86 = 0;
        LOBYTE(v39) = v38;
        v87 = v39;
        goto LABEL_31;
      }
    }
    else if ( sub_1F58D20((__int64)&v104) )
    {
      LOBYTE(v30) = sub_1F596B0((__int64)&v104);
      v28 = (char)v104;
      v87 = v30;
      v86 = v31;
      if ( !(_BYTE)v104 )
      {
        v32 = sub_1F58D30((__int64)&v104);
        goto LABEL_24;
      }
LABEL_31:
      v32 = word_432BB60[(unsigned __int8)(v28 - 14)];
LABEL_24:
      v33 = 0;
      v92 = v32;
      if ( v32 )
      {
        do
        {
          *(_QWORD *)&v35 = sub_1D38E70((__int64)a2, v33, (__int64)&v94, 0, a4, a5, a6);
          v22 = sub_1D332F0(
                  a2,
                  106,
                  (__int64)&v94,
                  (unsigned int)v87,
                  v86,
                  0,
                  *(double *)a4.m128i_i64,
                  a5,
                  a6,
                  v6,
                  v26,
                  v35);
          v23 = v36;
          v37 = (unsigned int)v108;
          if ( (unsigned int)v108 >= HIDWORD(v108) )
          {
            v82 = v36;
            v81 = v22;
            sub_16CD150((__int64)&v107, v109, 0, 16, (int)v22, (int)v36);
            v37 = (unsigned int)v108;
            v22 = v81;
            v23 = v82;
          }
          v34 = (__int64 **)&v107[2 * v37];
          ++v33;
          *v34 = v22;
          v34[1] = v23;
          LODWORD(v108) = v108 + 1;
        }
        while ( v92 != v33 );
      }
      goto LABEL_18;
    }
    v24 = (unsigned int)v108;
    if ( (unsigned int)v108 >= HIDWORD(v108) )
    {
      sub_16CD150((__int64)&v107, v109, 0, 16, (int)v22, (int)v23);
      v24 = (unsigned int)v108;
    }
    v25 = &v107[2 * v24];
    *v25 = v6;
    v25[1] = v26;
    LODWORD(v108) = v108 + 1;
LABEL_18:
    if ( ++v93 == v84 )
      break;
    v14 = *(_QWORD *)(a1 + 32);
  }
  v40 = *(char **)(a1 + 40);
  v18 = *v40;
  v41 = (const void **)*((_QWORD *)v40 + 1);
  LOBYTE(v96) = v18;
  v97 = v41;
  if ( !v18 )
  {
    v42 = sub_1F58D30((__int64)&v96);
    v43 = sub_1F596B0((__int64)&v96);
    v45 = v44;
    goto LABEL_38;
  }
LABEL_64:
  v76 = v18 - 14;
  v42 = word_432BB60[v76];
  switch ( v76 )
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
      v43 = 2;
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
      v43 = 3;
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
      v43 = 4;
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
      v43 = 5;
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
      v43 = 6;
      break;
    case 0x29u:
      v43 = 7;
      break;
    case 0x48u:
    case 0x49u:
    case 0x4Au:
    case 0x54u:
    case 0x55u:
    case 0x56u:
      v43 = 8;
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
      v43 = 9;
      break;
    case 0x50u:
    case 0x51u:
    case 0x52u:
    case 0x53u:
    case 0x5Cu:
    case 0x5Du:
    case 0x5Eu:
    case 0x5Fu:
      v43 = 10;
      break;
    default:
      ++*(_DWORD *)(v6 + 16);
      BUG();
  }
  v45 = 0;
LABEL_38:
  v46 = 0;
  v47 = v43;
  v48 = v103;
  v102 = 0x500000000LL;
  v101 = v103;
  if ( v42 )
  {
    v49 = 0;
    v50 = a2;
    v51 = v47;
    while ( 1 )
    {
      ++v49;
      v52 = &v48[16 * v46];
      *v52 = v51;
      v52[1] = v45;
      v46 = (unsigned int)(v102 + 1);
      LODWORD(v102) = v102 + 1;
      if ( v49 == v42 )
        break;
      if ( HIDWORD(v102) <= (unsigned int)v46 )
      {
        sub_16CD150((__int64)&v101, v103, 0, 16, (int)v22, (int)v23);
        v46 = (unsigned int)v102;
      }
      v48 = v101;
    }
    a2 = v50;
    v53 = v101;
  }
  else
  {
    v53 = v103;
  }
  v88 = v107;
  v90 = (unsigned int)v108;
  v54 = sub_1D25C30((__int64)a2, v53, v46);
  v57 = sub_1D23DE0(a2, v83, (__int64)&v94, v54, v55, v56, v88, v90);
  v104 = v106;
  v59 = v57;
  v105 = 0x800000000LL;
  if ( v42 )
  {
    v60 = a2;
    v61 = 0;
    v62 = 8;
    v63 = 0;
    v64 = v42;
    v65 = v59;
    while ( 1 )
    {
      v66 = v63;
      if ( v62 <= (unsigned int)v61 )
      {
        sub_16CD150((__int64)&v104, v106, 0, 16, v59, v58);
        v61 = (unsigned int)v105;
      }
      v67 = (__int64 *)&v104[16 * v61];
      ++v63;
      *v67 = v65;
      v67[1] = v66;
      v61 = (unsigned int)(v105 + 1);
      LODWORD(v105) = v105 + 1;
      if ( v63 == v64 )
        break;
      v62 = HIDWORD(v105);
    }
    a2 = v60;
    v68 = v104;
  }
  else
  {
    v61 = 0;
    v68 = v106;
  }
  *((_QWORD *)&v78 + 1) = v61;
  *(_QWORD *)&v78 = v68;
  v70 = sub_1D359D0(a2, 104, (__int64)&v94, v96, v97, 0, *(double *)a4.m128i_i64, a5, a6, v78);
  v71 = v69;
  v72 = *(unsigned int *)(a3 + 8);
  if ( (unsigned int)v72 >= *(_DWORD *)(a3 + 12) )
  {
    v89 = v70;
    v91 = v69;
    sub_16CD150(a3, (const void *)(a3 + 16), 0, 16, (int)v70, (int)v69);
    v72 = *(unsigned int *)(a3 + 8);
    v70 = v89;
    v71 = v91;
  }
  v73 = (__int64 **)(*(_QWORD *)a3 + 16 * v72);
  *v73 = v70;
  v73[1] = v71;
  v74 = v104;
  ++*(_DWORD *)(a3 + 8);
  if ( v74 != v106 )
    _libc_free((unsigned __int64)v74);
  if ( v101 != v103 )
    _libc_free((unsigned __int64)v101);
  if ( v107 != v109 )
    _libc_free((unsigned __int64)v107);
  if ( v94 )
    sub_161E7C0((__int64)&v94, v94);
}
