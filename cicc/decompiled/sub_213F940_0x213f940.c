// Function: sub_213F940
// Address: 0x213f940
//
__int64 __fastcall sub_213F940(__int64 a1, __int64 a2, __m128i a3, double a4, __m128i a5)
{
  unsigned __int64 *v7; // rax
  __int64 v8; // rsi
  unsigned int v9; // ecx
  __int64 v10; // rsi
  unsigned __int8 *v11; // rdx
  __int64 v12; // r12
  __int64 v13; // r13
  __int64 v14; // r12
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // r13
  __int64 v19; // r12
  __int64 v20; // rax
  char v21; // dl
  __int64 v22; // rax
  __int64 v23; // rax
  unsigned __int8 v24; // cl
  __int64 v25; // rax
  _QWORD *v26; // rbx
  __int64 v27; // rax
  const void **v28; // r8
  __int64 v29; // rbx
  __int64 v30; // r12
  __int64 v31; // rdx
  __int64 v32; // r13
  __int64 v33; // r11
  char v34; // al
  __int64 v35; // rdx
  unsigned int v36; // ebx
  unsigned int v37; // r12d
  unsigned __int8 v38; // bl
  unsigned int v39; // r13d
  const void **v40; // rbx
  unsigned int v41; // eax
  unsigned int v42; // r13d
  __int64 v43; // rax
  __int64 *v44; // rdi
  int v45; // edx
  __int64 v46; // rax
  __int64 *v47; // rdi
  int v48; // edx
  unsigned int v49; // edx
  unsigned __int8 v50; // al
  __int64 v51; // rdx
  unsigned int v52; // ebx
  unsigned int v53; // eax
  const void **v54; // r8
  unsigned int v55; // ebx
  __int64 v56; // r12
  unsigned __int64 v57; // rdx
  unsigned __int64 v58; // r13
  __int64 (__fastcall *v59)(__int64, __int64); // rbx
  __int64 v60; // rax
  __int64 v61; // rdx
  unsigned int v62; // eax
  __int128 v63; // rax
  unsigned __int8 v64; // bl
  bool v65; // al
  char v66; // al
  __int64 v67; // rdx
  char v68; // al
  const void **v69; // rdx
  const void **v70; // rdx
  const void **v71; // rdx
  const void **v72; // rdx
  __int128 v73; // [rsp-10h] [rbp-120h]
  __int128 v74; // [rsp-10h] [rbp-120h]
  __int128 v75; // [rsp-10h] [rbp-120h]
  __int64 v76; // [rsp-10h] [rbp-120h]
  __int64 v77; // [rsp-8h] [rbp-118h]
  unsigned int v78; // [rsp+0h] [rbp-110h]
  unsigned int v79; // [rsp+8h] [rbp-108h]
  __int64 v80; // [rsp+10h] [rbp-100h]
  __int64 v81; // [rsp+18h] [rbp-F8h]
  unsigned __int64 v82; // [rsp+20h] [rbp-F0h]
  __int64 v83; // [rsp+28h] [rbp-E8h]
  __int64 v84; // [rsp+30h] [rbp-E0h]
  __int64 v85; // [rsp+30h] [rbp-E0h]
  __int64 v86; // [rsp+30h] [rbp-E0h]
  unsigned int v87; // [rsp+3Ch] [rbp-D4h]
  unsigned __int8 v88; // [rsp+3Ch] [rbp-D4h]
  unsigned __int8 v89; // [rsp+3Ch] [rbp-D4h]
  unsigned __int64 v90; // [rsp+40h] [rbp-D0h]
  unsigned int v91; // [rsp+40h] [rbp-D0h]
  const void **v92; // [rsp+40h] [rbp-D0h]
  unsigned int v93; // [rsp+48h] [rbp-C8h]
  _QWORD *v94; // [rsp+48h] [rbp-C8h]
  _QWORD *v95; // [rsp+48h] [rbp-C8h]
  __int64 v96; // [rsp+48h] [rbp-C8h]
  unsigned int v97; // [rsp+80h] [rbp-90h] BYREF
  const void **v98; // [rsp+88h] [rbp-88h]
  __int64 v99; // [rsp+90h] [rbp-80h] BYREF
  int v100; // [rsp+98h] [rbp-78h]
  char v101[8]; // [rsp+A0h] [rbp-70h] BYREF
  __int64 v102; // [rsp+A8h] [rbp-68h]
  __int128 v103; // [rsp+B0h] [rbp-60h] BYREF
  __int128 v104; // [rsp+C0h] [rbp-50h] BYREF
  const void **v105; // [rsp+D0h] [rbp-40h]

  sub_1F40D10(
    (__int64)&v104,
    *(_QWORD *)a1,
    *(_QWORD *)(*(_QWORD *)(a1 + 8) + 48LL),
    **(unsigned __int8 **)(a2 + 40),
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  LOBYTE(v97) = BYTE8(v104);
  v98 = v105;
  v7 = *(unsigned __int64 **)(a2 + 32);
  v82 = *v7;
  v90 = *v7;
  v8 = *(_QWORD *)(a2 + 72);
  v81 = v7[1];
  v9 = *((_DWORD *)v7 + 2);
  v99 = v8;
  if ( v8 )
  {
    v87 = v9;
    sub_1623A60((__int64)&v99, v8, 2);
    v9 = v87;
  }
  v10 = *(_QWORD *)a1;
  v100 = *(_DWORD *)(a2 + 64);
  v83 = v9;
  v11 = (unsigned __int8 *)(*(_QWORD *)(v90 + 40) + 16LL * v9);
  v84 = 16LL * v9;
  sub_1F40D10((__int64)&v104, v10, *(_QWORD *)(*(_QWORD *)(a1 + 8) + 48LL), *v11, *((_QWORD *)v11 + 1));
  v88 = v104;
  switch ( (char)v104 )
  {
    case 0:
    case 2:
      v12 = v90;
      v13 = v83;
      goto LABEL_6;
    case 1:
      v12 = sub_2138AD0(a1, v82, v81);
      v13 = v49;
LABEL_6:
      *((_QWORD *)&v73 + 1) = v13;
      *(_QWORD *)&v73 = v12;
      v14 = sub_1D309E0(
              *(__int64 **)(a1 + 8),
              145,
              (__int64)&v99,
              v97,
              v98,
              0,
              *(double *)a3.m128i_i64,
              a4,
              *(double *)a5.m128i_i64,
              v73);
      goto LABEL_7;
    case 3:
    case 4:
    case 5:
    case 7:
      v16 = sub_20363F0(a1, v82, v81);
      v18 = v17;
      v19 = v16;
      v20 = *(_QWORD *)(v16 + 40) + 16LL * (unsigned int)v17;
      v21 = *(_BYTE *)v20;
      v22 = *(_QWORD *)(v20 + 8);
      LOBYTE(v104) = v21;
      *((_QWORD *)&v104 + 1) = v22;
      if ( v21 )
        v91 = word_4310720[(unsigned __int8)(v21 - 14)];
      else
        v91 = sub_1F58D30((__int64)&v104);
      v23 = *(_QWORD *)(a2 + 40);
      v24 = *(_BYTE *)v23;
      v25 = *(_QWORD *)(v23 + 8);
      LOBYTE(v104) = v24;
      v85 = v25;
      *((_QWORD *)&v104 + 1) = v25;
      if ( v24 )
      {
        if ( (unsigned __int8)(v24 - 14) <= 0x5Fu )
        {
          switch ( v24 )
          {
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
              v24 = 3;
              break;
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
              v24 = 4;
              break;
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
              v24 = 5;
              break;
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
              v24 = 6;
              break;
            case 0x37u:
              v24 = v88;
              break;
            case 0x56u:
            case 0x57u:
            case 0x58u:
            case 0x62u:
            case 0x63u:
            case 0x64u:
              v24 = 8;
              break;
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
              v24 = 9;
              break;
            case 0x5Eu:
            case 0x5Fu:
            case 0x60u:
            case 0x61u:
            case 0x6Au:
            case 0x6Bu:
            case 0x6Cu:
            case 0x6Du:
              v24 = 10;
              break;
            default:
              v24 = 2;
              break;
          }
          v85 = 0;
        }
      }
      else
      {
        v65 = sub_1F58D20((__int64)&v104);
        v24 = 0;
        if ( v65 )
        {
          v66 = sub_1F596B0((__int64)&v104);
          v85 = v67;
          v24 = v66;
        }
      }
      v93 = v24;
      v26 = *(_QWORD **)(*(_QWORD *)(a1 + 8) + 48LL);
      LOBYTE(v27) = sub_1D15020(v24, v91);
      v28 = 0;
      if ( !(_BYTE)v27 )
      {
        v27 = sub_1F593D0(v26, v93, v85, v91);
        v80 = v27;
        v28 = v72;
      }
      v29 = v80;
      *((_QWORD *)&v74 + 1) = v18;
      *(_QWORD *)&v74 = v19;
      LOBYTE(v29) = v27;
      v30 = sub_1D309E0(
              *(__int64 **)(a1 + 8),
              145,
              (__int64)&v99,
              (unsigned int)v29,
              v28,
              0,
              *(double *)a3.m128i_i64,
              a4,
              *(double *)a5.m128i_i64,
              v74);
      v32 = v31;
      if ( (_BYTE)v97 )
      {
        switch ( (char)v97 )
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
            v50 = 2;
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
            v50 = 3;
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
            v50 = 4;
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
            v50 = 5;
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
            v50 = 6;
            break;
          case 55:
            v50 = v88;
            break;
          case 86:
          case 87:
          case 88:
          case 98:
          case 99:
          case 100:
            v50 = 8;
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
            v50 = 9;
            break;
          case 94:
          case 95:
          case 96:
          case 97:
          case 106:
          case 107:
          case 108:
          case 109:
            v50 = 10;
            break;
          default:
            *(_DWORD *)(v29 + 8) = (2 * (*(_DWORD *)(v29 + 8) >> 1) + 2) | *(_DWORD *)(v29 + 8) & 1;
            BUG();
        }
        v86 = 0;
      }
      else
      {
        v50 = sub_1F596B0((__int64)&v97);
        v86 = v51;
      }
      v95 = *(_QWORD **)(*(_QWORD *)(a1 + 8) + 48LL);
      v52 = v50;
      LOBYTE(v53) = sub_1D15020(v50, v91);
      v54 = 0;
      if ( !(_BYTE)v53 )
      {
        v53 = sub_1F593D0(v95, v52, v86, v91);
        v79 = v53;
        v54 = v70;
      }
      v55 = v79;
      *((_QWORD *)&v75 + 1) = v32;
      *(_QWORD *)&v75 = v30;
      LOBYTE(v55) = v53;
      v56 = sub_1D309E0(
              *(__int64 **)(a1 + 8),
              143,
              (__int64)&v99,
              v55,
              v54,
              0,
              *(double *)a3.m128i_i64,
              a4,
              *(double *)a5.m128i_i64,
              v75);
      v58 = v57;
      v96 = *(_QWORD *)a1;
      v59 = *(__int64 (__fastcall **)(__int64, __int64))(**(_QWORD **)a1 + 48LL);
      v60 = sub_1E0A0C0(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 32LL));
      if ( v59 != sub_1D13A20 )
        return sub_2140100(v60, v77, v61, v76, v96);
      v62 = 8 * sub_15A9520(v60, 0);
      if ( v62 == 32 )
      {
        v89 = 5;
      }
      else if ( v62 > 0x20 )
      {
        if ( v62 == 64 )
        {
          v89 = 6;
        }
        else
        {
          v64 = v88;
          if ( v62 != 128 )
            v64 = 0;
          v89 = v64;
        }
      }
      else
      {
        v89 = 3;
        if ( v62 != 8 )
          v89 = 4 * (v62 == 16);
      }
      *(_QWORD *)&v63 = sub_1D38BB0(*(_QWORD *)(a1 + 8), 0, (__int64)&v99, v89, 0, 0, a3, a4, a5, 0);
      v14 = (__int64)sub_1D332F0(
                       *(__int64 **)(a1 + 8),
                       109,
                       (__int64)&v99,
                       v97,
                       v98,
                       0,
                       *(double *)a3.m128i_i64,
                       a4,
                       a5,
                       v56,
                       v58,
                       v63);
      goto LABEL_7;
    case 6:
      v33 = *(_QWORD *)(v90 + 40) + v84;
      v34 = *(_BYTE *)v33;
      v35 = *(_QWORD *)(v33 + 8);
      v101[0] = v34;
      v102 = v35;
      if ( v34 )
        v36 = word_4310720[(unsigned __int8)(v34 - 14)];
      else
        v36 = sub_1F58D30((__int64)v101);
      *(_QWORD *)&v103 = 0;
      v37 = v36 >> 1;
      DWORD2(v103) = 0;
      *(_QWORD *)&v104 = 0;
      DWORD2(v104) = 0;
      sub_2017DE0(a1, v82, v81, &v103, &v104);
      v38 = v97;
      if ( (_BYTE)v97 )
      {
        if ( (unsigned __int8)(v97 - 14) > 0x5Fu )
        {
LABEL_22:
          v92 = v98;
          goto LABEL_23;
        }
        switch ( (char)v97 )
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
            v38 = v88;
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
        v92 = 0;
      }
      else
      {
        if ( !sub_1F58D20((__int64)&v97) )
          goto LABEL_22;
        v68 = sub_1F596B0((__int64)&v97);
        v92 = v69;
        v38 = v68;
      }
LABEL_23:
      v39 = v38;
      v40 = 0;
      v94 = *(_QWORD **)(*(_QWORD *)(a1 + 8) + 48LL);
      LOBYTE(v41) = sub_1D15020(v39, v37);
      if ( !(_BYTE)v41 )
      {
        v41 = sub_1F593D0(v94, v39, (__int64)v92, v37);
        v78 = v41;
        v40 = v71;
      }
      v42 = v78;
      LOBYTE(v42) = v41;
      v43 = sub_1D309E0(
              *(__int64 **)(a1 + 8),
              145,
              (__int64)&v99,
              v42,
              v40,
              0,
              *(double *)a3.m128i_i64,
              a4,
              *(double *)a5.m128i_i64,
              v103);
      v44 = *(__int64 **)(a1 + 8);
      *(_QWORD *)&v103 = v43;
      DWORD2(v103) = v45;
      v46 = sub_1D309E0(
              v44,
              145,
              (__int64)&v99,
              v42,
              v40,
              0,
              *(double *)a3.m128i_i64,
              a4,
              *(double *)a5.m128i_i64,
              v104);
      v47 = *(__int64 **)(a1 + 8);
      *(_QWORD *)&v104 = v46;
      DWORD2(v104) = v48;
      v14 = (__int64)sub_1D332F0(
                       v47,
                       107,
                       (__int64)&v99,
                       v97,
                       v98,
                       0,
                       *(double *)a3.m128i_i64,
                       a4,
                       a5,
                       v103,
                       *((unsigned __int64 *)&v103 + 1),
                       __PAIR128__(*((unsigned __int64 *)&v104 + 1), v46));
LABEL_7:
      if ( v99 )
        sub_161E7C0((__int64)&v99, v99);
      return v14;
  }
}
