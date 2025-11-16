// Function: sub_380B540
// Address: 0x380b540
//
__int64 __fastcall sub_380B540(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, int a6, __m128i a7)
{
  unsigned int v7; // ebx
  unsigned __int8 *v8; // rax
  __int64 v9; // rdx
  __int64 result; // rax
  unsigned __int64 v11; // rcx
  unsigned int v12; // edx
  __int64 v13; // r8
  unsigned int v14; // edx
  int v15; // edx
  __int16 v16; // ax
  unsigned int v17; // edx
  __int16 v18; // ax
  unsigned int v19; // edx
  unsigned int v20; // edx
  unsigned int v21; // edx
  unsigned int v22; // edx
  __int16 v23; // ax
  __int16 v24; // ax
  __int16 v25; // ax
  __int16 v26; // ax
  __int16 v27; // ax
  __int16 v28; // ax
  __int16 v29; // ax
  int v30; // edx
  __int16 v31; // ax
  unsigned int v32; // edx
  __int16 v33; // ax
  __int16 v34; // ax
  __int16 v35; // ax
  __int16 v36; // ax
  __int16 v37; // ax
  __int16 v38; // ax
  __int16 v39; // ax
  __int16 v40; // ax
  __int16 v41; // ax
  __int16 v42; // ax
  __int16 v43; // ax
  __int16 v44; // ax
  __int16 v45; // ax
  __int16 v46; // ax
  __int16 v47; // ax
  __int16 v48; // ax
  __int16 v49; // ax
  __int16 v50; // ax
  unsigned int v51; // edx
  __int16 v52; // ax
  unsigned int v53; // edx
  unsigned int v54; // edx
  unsigned int v55; // edx
  unsigned int v56; // edx
  unsigned int v57; // edx
  __int64 v58; // rax
  unsigned int v59; // edx
  unsigned int v60; // edx
  unsigned int v61; // edx
  unsigned int v62; // edx
  unsigned int v63; // edx
  unsigned int v64; // edx
  unsigned int v65; // edx
  unsigned int v66; // edx
  unsigned int v67; // edx
  unsigned int v68; // edx
  unsigned int v69; // edx
  unsigned int v70; // edx
  unsigned int v71; // edx
  int v72; // eax
  unsigned int v73; // edx
  __int16 v74; // ax
  __int16 v75; // ax
  __int16 v76; // ax
  __int16 v77; // ax
  __int16 v78; // ax
  unsigned int v79; // edx
  __int16 v80; // ax
  unsigned int v81; // edx
  unsigned int v82; // edx
  unsigned int v83; // edx
  __int64 v84; // [rsp+1A8h] [rbp-28h]

  v7 = a3;
  switch ( *(_DWORD *)(a2 + 24) )
  {
    case 0xC:
      result = (__int64)sub_37FC120((__int64 *)a1, a2, a7);
      v11 = result;
      v13 = v83;
      goto LABEL_6;
    case 0x33:
      result = (__int64)sub_37FD3E0((__int64 *)a1, a2);
      v11 = result;
      v13 = v82;
      goto LABEL_6;
    case 0x34:
      result = (__int64)sub_3806620((__int64 *)a1, a2, a7);
      v11 = result;
      v13 = v81;
      goto LABEL_6;
    case 0x35:
      result = (__int64)sub_37FC450(a1, a2, a7);
      v11 = result;
      v13 = v66;
      goto LABEL_6;
    case 0x36:
      result = (__int64)sub_37FBFD0((__int64 *)a1, a2, a7);
      v11 = result;
      v13 = v61;
      goto LABEL_6;
    case 0x37:
      v58 = sub_3761980(a1, a2, a3);
      result = (__int64)sub_375A6A0(a1, v58, v59, a7);
      v11 = result;
      v13 = v60;
      goto LABEL_6;
    case 0x60:
    case 0x65:
      v30 = 55;
      v44 = **(_WORD **)(a2 + 48);
      if ( v44 != 12 )
      {
        v30 = 56;
        if ( v44 != 13 )
        {
          v30 = 57;
          if ( v44 != 14 )
          {
            v30 = 58;
            if ( v44 != 15 )
            {
              v30 = 729;
              if ( v44 == 16 )
                v30 = 59;
            }
          }
        }
      }
      goto LABEL_83;
    case 0x61:
    case 0x66:
      v30 = 60;
      v43 = **(_WORD **)(a2 + 48);
      if ( v43 != 12 )
      {
        v30 = 61;
        if ( v43 != 13 )
        {
          v30 = 62;
          if ( v43 != 14 )
          {
            v30 = 63;
            if ( v43 != 15 )
            {
              v30 = 729;
              if ( v43 == 16 )
                v30 = 64;
            }
          }
        }
      }
      goto LABEL_83;
    case 0x62:
    case 0x67:
      v30 = 65;
      v42 = **(_WORD **)(a2 + 48);
      if ( v42 != 12 )
      {
        v30 = 66;
        if ( v42 != 13 )
        {
          v30 = 67;
          if ( v42 != 14 )
          {
            v30 = 68;
            if ( v42 != 15 )
            {
              v30 = 729;
              if ( v42 == 16 )
                v30 = 69;
            }
          }
        }
      }
      goto LABEL_83;
    case 0x63:
    case 0x68:
      v30 = 70;
      v41 = **(_WORD **)(a2 + 48);
      if ( v41 != 12 )
      {
        v30 = 71;
        if ( v41 != 13 )
        {
          v30 = 72;
          if ( v41 != 14 )
          {
            v30 = 73;
            if ( v41 != 15 )
            {
              v30 = 729;
              if ( v41 == 16 )
                v30 = 74;
            }
          }
        }
      }
      goto LABEL_83;
    case 0x64:
    case 0x69:
      v30 = 75;
      v40 = **(_WORD **)(a2 + 48);
      if ( v40 != 12 )
      {
        v30 = 76;
        if ( v40 != 13 )
        {
          v30 = 77;
          if ( v40 != 14 )
          {
            v30 = 78;
            if ( v40 != 15 )
            {
              v30 = 729;
              if ( v40 == 16 )
                v30 = 79;
            }
          }
        }
      }
      goto LABEL_83;
    case 0x6A:
    case 0x96:
      result = sub_3807360((__int64 *)a1, a2);
      v11 = result;
      v13 = v51;
      goto LABEL_6;
    case 0x6B:
    case 0xF6:
      v15 = 90;
      v50 = **(_WORD **)(a2 + 48);
      if ( v50 != 12 )
      {
        v15 = 91;
        if ( v50 != 13 )
        {
          v15 = 92;
          if ( v50 != 14 )
          {
            v15 = 93;
            if ( v50 != 15 )
            {
              v15 = 729;
              if ( v50 == 16 )
                v15 = 94;
            }
          }
        }
      }
      goto LABEL_15;
    case 0x6C:
    case 0x101:
      v30 = 212;
      v49 = **(_WORD **)(a2 + 48);
      if ( v49 != 12 )
      {
        v30 = 213;
        if ( v49 != 13 )
        {
          v30 = 214;
          if ( v49 != 14 )
          {
            v30 = 215;
            if ( v49 != 15 )
            {
              v30 = 729;
              if ( v49 == 16 )
                v30 = 216;
            }
          }
        }
      }
      goto LABEL_83;
    case 0x6D:
    case 0x6E:
    case 0x102:
    case 0x103:
      result = (__int64)sub_38079C0((__int64 *)a1, a2);
      v11 = result;
      v13 = v14;
      goto LABEL_6;
    case 0x6F:
    case 0xF8:
      v15 = 155;
      v48 = **(_WORD **)(a2 + 48);
      if ( v48 != 12 )
      {
        v15 = 156;
        if ( v48 != 13 )
        {
          v15 = 157;
          if ( v48 != 14 )
          {
            v15 = 158;
            if ( v48 != 15 )
            {
              v15 = 729;
              if ( v48 == 16 )
                v15 = 159;
            }
          }
        }
      }
      goto LABEL_15;
    case 0x70:
    case 0xF9:
      v15 = 160;
      v37 = **(_WORD **)(a2 + 48);
      if ( v37 != 12 )
      {
        v15 = 161;
        if ( v37 != 13 )
        {
          v15 = 162;
          if ( v37 != 14 )
          {
            v15 = 163;
            if ( v37 != 15 )
            {
              v15 = 729;
              if ( v37 == 16 )
                v15 = 164;
            }
          }
        }
      }
      goto LABEL_15;
    case 0x71:
    case 0xFA:
      v15 = 165;
      v36 = **(_WORD **)(a2 + 48);
      if ( v36 != 12 )
      {
        v15 = 166;
        if ( v36 != 13 )
        {
          v15 = 167;
          if ( v36 != 14 )
          {
            v15 = 168;
            if ( v36 != 15 )
            {
              v15 = 729;
              if ( v36 == 16 )
                v15 = 169;
            }
          }
        }
      }
      goto LABEL_15;
    case 0x72:
    case 0xFB:
      v15 = 185;
      v35 = **(_WORD **)(a2 + 48);
      if ( v35 != 12 )
      {
        v15 = 186;
        if ( v35 != 13 )
        {
          v15 = 187;
          if ( v35 != 14 )
          {
            v15 = 188;
            if ( v35 != 15 )
            {
              v15 = 729;
              if ( v35 == 16 )
                v15 = 189;
            }
          }
        }
      }
      goto LABEL_15;
    case 0x73:
    case 0xFC:
      v15 = 190;
      v34 = **(_WORD **)(a2 + 48);
      if ( v34 != 12 )
      {
        v15 = 191;
        if ( v34 != 13 )
        {
          v15 = 192;
          if ( v34 != 14 )
          {
            v15 = 193;
            if ( v34 != 15 )
            {
              v15 = 729;
              if ( v34 == 16 )
                v15 = 194;
            }
          }
        }
      }
      goto LABEL_15;
    case 0x74:
    case 0xFD:
      v15 = 195;
      v33 = **(_WORD **)(a2 + 48);
      if ( v33 != 12 )
      {
        v15 = 196;
        if ( v33 != 13 )
        {
          v15 = 197;
          if ( v33 != 14 )
          {
            v15 = 198;
            if ( v33 != 15 )
            {
              v15 = 729;
              if ( v33 == 16 )
                v15 = 199;
            }
          }
        }
      }
      goto LABEL_15;
    case 0x75:
    case 0x104:
      v30 = 200;
      v31 = **(_WORD **)(a2 + 48);
      if ( v31 != 12 )
      {
        v30 = 201;
        if ( v31 != 13 )
        {
          v30 = 202;
          if ( v31 != 14 )
          {
            v30 = 203;
            if ( v31 != 15 )
            {
              v30 = 729;
              if ( v31 == 16 )
                v30 = 204;
            }
          }
        }
      }
      goto LABEL_83;
    case 0x76:
    case 0xFE:
      v15 = 170;
      v29 = **(_WORD **)(a2 + 48);
      if ( v29 != 12 )
      {
        v15 = 171;
        if ( v29 != 13 )
        {
          v15 = 172;
          if ( v29 != 14 )
          {
            v15 = 173;
            if ( v29 != 15 )
            {
              v15 = 729;
              if ( v29 == 16 )
                v15 = 174;
            }
          }
        }
      }
      goto LABEL_15;
    case 0x77:
    case 0xFF:
      v15 = 175;
      v28 = **(_WORD **)(a2 + 48);
      if ( v28 != 12 )
      {
        v15 = 176;
        if ( v28 != 13 )
        {
          v15 = 177;
          if ( v28 != 14 )
          {
            v15 = 178;
            if ( v28 != 15 )
            {
              v15 = 729;
              if ( v28 == 16 )
                v15 = 179;
            }
          }
        }
      }
      goto LABEL_15;
    case 0x78:
    case 0x100:
      v15 = 180;
      v27 = **(_WORD **)(a2 + 48);
      if ( v27 != 12 )
      {
        v15 = 181;
        if ( v27 != 13 )
        {
          v15 = 182;
          if ( v27 != 14 )
          {
            v15 = 183;
            if ( v27 != 15 )
            {
              v15 = 729;
              if ( v27 == 16 )
                v15 = 184;
            }
          }
        }
      }
      goto LABEL_15;
    case 0x79:
    case 0x109:
      v15 = 130;
      v26 = **(_WORD **)(a2 + 48);
      if ( v26 != 12 )
      {
        v15 = 131;
        if ( v26 != 13 )
        {
          v15 = 132;
          if ( v26 != 14 )
          {
            v15 = 133;
            if ( v26 != 15 )
            {
              v15 = 729;
              if ( v26 == 16 )
                v15 = 134;
            }
          }
        }
      }
      goto LABEL_15;
    case 0x7A:
    case 0x10A:
      v15 = 140;
      v25 = **(_WORD **)(a2 + 48);
      if ( v25 != 12 )
      {
        v15 = 141;
        if ( v25 != 13 )
        {
          v15 = 142;
          if ( v25 != 14 )
          {
            v15 = 143;
            if ( v25 != 15 )
            {
              v15 = 729;
              if ( v25 == 16 )
                v15 = 144;
            }
          }
        }
      }
      goto LABEL_15;
    case 0x7B:
    case 0x106:
      v15 = 100;
      v24 = **(_WORD **)(a2 + 48);
      if ( v24 != 12 )
      {
        v15 = 101;
        if ( v24 != 13 )
        {
          v15 = 102;
          if ( v24 != 14 )
          {
            v15 = 103;
            if ( v24 != 15 )
            {
              v15 = 729;
              if ( v24 == 16 )
                v15 = 104;
            }
          }
        }
      }
      goto LABEL_15;
    case 0x7C:
    case 0x108:
      v15 = 120;
      v23 = **(_WORD **)(a2 + 48);
      if ( v23 != 12 )
      {
        v15 = 121;
        if ( v23 != 13 )
        {
          v15 = 122;
          if ( v23 != 14 )
          {
            v15 = 123;
            if ( v23 != 15 )
            {
              v15 = 729;
              if ( v23 == 16 )
                v15 = 124;
            }
          }
        }
      }
      goto LABEL_15;
    case 0x7D:
    case 0x107:
      v15 = 110;
      v47 = **(_WORD **)(a2 + 48);
      if ( v47 != 12 )
      {
        v15 = 111;
        if ( v47 != 13 )
        {
          v15 = 112;
          if ( v47 != 14 )
          {
            v15 = 113;
            if ( v47 != 15 )
            {
              v15 = 729;
              if ( v47 == 16 )
                v15 = 114;
            }
          }
        }
      }
      goto LABEL_15;
    case 0x7E:
    case 0x10E:
      v15 = 232;
      v46 = **(_WORD **)(a2 + 48);
      if ( v46 != 12 )
      {
        v15 = 233;
        if ( v46 != 13 )
        {
          v15 = 234;
          if ( v46 != 14 )
          {
            v15 = 235;
            if ( v46 != 15 )
            {
              v15 = 729;
              if ( v46 == 16 )
                v15 = 236;
            }
          }
        }
      }
      goto LABEL_15;
    case 0x7F:
    case 0x10F:
      v15 = 237;
      v45 = **(_WORD **)(a2 + 48);
      if ( v45 != 12 )
      {
        v15 = 238;
        if ( v45 != 13 )
        {
          v15 = 239;
          if ( v45 != 14 )
          {
            v15 = 240;
            if ( v45 != 15 )
            {
              v15 = 729;
              if ( v45 == 16 )
                v15 = 241;
            }
          }
        }
      }
      goto LABEL_15;
    case 0x80:
    case 0x118:
      result = (__int64)sub_3808C80(a1, a2);
      v11 = result;
      v13 = v20;
      goto LABEL_6;
    case 0x81:
    case 0x117:
      result = (__int64)sub_3808BF0(a1, a2);
      v11 = result;
      v13 = v19;
      goto LABEL_6;
    case 0x82:
    case 0x10C:
      v15 = 222;
      v18 = **(_WORD **)(a2 + 48);
      if ( v18 != 12 )
      {
        v15 = 223;
        if ( v18 != 13 )
        {
          v15 = 224;
          if ( v18 != 14 )
          {
            v15 = 225;
            if ( v18 != 15 )
            {
              v15 = 729;
              if ( v18 == 16 )
                v15 = 226;
            }
          }
        }
      }
      goto LABEL_15;
    case 0x83:
    case 0x112:
      v15 = 252;
      v16 = **(_WORD **)(a2 + 48);
      if ( v16 != 12 )
      {
        v15 = 253;
        if ( v16 != 13 )
        {
          v15 = 254;
          if ( v16 != 14 )
          {
            v15 = 255;
            if ( v16 != 15 )
            {
              v15 = 729;
              if ( v16 == 16 )
                v15 = 256;
            }
          }
        }
      }
      goto LABEL_15;
    case 0x84:
    case 0x110:
      v15 = 242;
      v39 = **(_WORD **)(a2 + 48);
      if ( v39 != 12 )
      {
        v15 = 243;
        if ( v39 != 13 )
        {
          v15 = 244;
          if ( v39 != 14 )
          {
            v15 = 245;
            if ( v39 != 15 )
            {
              v15 = 729;
              if ( v39 == 16 )
                v15 = 246;
            }
          }
        }
      }
      goto LABEL_15;
    case 0x85:
    case 0x111:
      v15 = 247;
      v38 = **(_WORD **)(a2 + 48);
      if ( v38 != 12 )
      {
        v15 = 248;
        if ( v38 != 13 )
        {
          v15 = 249;
          if ( v38 != 14 )
          {
            v15 = 250;
            if ( v38 != 15 )
            {
              v15 = 729;
              if ( v38 == 16 )
                v15 = 251;
            }
          }
        }
      }
      goto LABEL_15;
    case 0x86:
    case 0x10D:
      v15 = 227;
      v52 = **(_WORD **)(a2 + 48);
      if ( v52 != 12 )
      {
        v15 = 228;
        if ( v52 != 13 )
        {
          v15 = 229;
          if ( v52 != 14 )
          {
            v15 = 230;
            if ( v52 != 15 )
            {
              v15 = 729;
              if ( v52 == 16 )
                v15 = 231;
            }
          }
        }
      }
      goto LABEL_15;
    case 0x8F:
    case 0x90:
    case 0xDC:
    case 0xDD:
      result = sub_37FD620(a1, a2, a7, a3, a4, a5, a6);
      v11 = result;
      v13 = v12;
      goto LABEL_6;
    case 0x91:
    case 0xE6:
      result = sub_37FCC90((__int64 *)a1, a2);
      v11 = result;
      v13 = v22;
      goto LABEL_6;
    case 0x92:
    case 0xE9:
      result = (__int64)sub_380AEA0((__int64 *)a1, a2, a7);
      v11 = result;
      v13 = v21;
      goto LABEL_6;
    case 0x98:
      result = (__int64)sub_3806B30((__int64 *)a1, a2, a7);
      v11 = result;
      v13 = v63;
      goto LABEL_6;
    case 0x9E:
      result = (__int64)sub_37FC640(a1, a2, a7);
      v11 = result;
      v13 = v62;
      goto LABEL_6;
    case 0xCD:
      result = sub_3808940(a1, a2);
      v11 = result;
      v13 = v57;
      goto LABEL_6;
    case 0xCF:
      result = (__int64)sub_3808AD0(a1, a2);
      v11 = result;
      v13 = v56;
      goto LABEL_6;
    case 0xEA:
      result = (__int64)sub_375A6A0(a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL), a7);
      v11 = result;
      v13 = v55;
      goto LABEL_6;
    case 0xEC:
      result = sub_37FC770((__int64 *)a1, a2);
      v11 = result;
      v13 = v64;
      goto LABEL_6;
    case 0xF0:
      result = (__int64)sub_37FCAF0((__int64 *)a1, a2, a7);
      v11 = result;
      v13 = v54;
      goto LABEL_6;
    case 0xF4:
      result = (__int64)sub_3807760((__int64 *)a1, a2, a7);
      v11 = result;
      v13 = v53;
      goto LABEL_6;
    case 0xF5:
      result = (__int64)sub_3806880((__int64 *)a1, a2, a7);
      v11 = result;
      v13 = v67;
      goto LABEL_6;
    case 0xF7:
      v15 = 95;
      v80 = **(_WORD **)(a2 + 48);
      if ( v80 != 12 )
      {
        v15 = 96;
        if ( v80 != 13 )
        {
          v15 = 97;
          if ( v80 != 14 )
          {
            v15 = 98;
            if ( v80 != 15 )
            {
              v15 = 729;
              if ( v80 == 16 )
                v15 = 99;
            }
          }
        }
      }
      goto LABEL_15;
    case 0x105:
      result = (__int64)sub_3807F10((__int64 *)a1, a2);
      v11 = result;
      v13 = v79;
      goto LABEL_6;
    case 0x10B:
      v15 = 150;
      v78 = **(_WORD **)(a2 + 48);
      if ( v78 != 12 )
      {
        v15 = 151;
        if ( v78 != 13 )
        {
          v15 = 152;
          if ( v78 != 14 )
          {
            v15 = 153;
            if ( v78 != 15 )
            {
              v15 = 729;
              if ( v78 == 16 )
                v15 = 154;
            }
          }
        }
      }
LABEL_15:
      result = sub_3806040((__int64 *)a1, a2, v15);
      v11 = result;
      v13 = v17;
      if ( !result )
        return result;
      return sub_375F330(a1, a2, v7, v11, v13);
    case 0x11B:
      v30 = 272;
      v77 = **(_WORD **)(a2 + 48);
      if ( v77 != 12 )
      {
        v30 = 273;
        if ( v77 != 13 )
        {
          v30 = 274;
          if ( v77 != 14 )
          {
            v30 = 275;
            if ( v77 != 15 )
            {
              v30 = 729;
              if ( v77 == 16 )
                v30 = 276;
            }
          }
        }
      }
      goto LABEL_83;
    case 0x11C:
      v30 = 277;
      v76 = **(_WORD **)(a2 + 48);
      if ( v76 != 12 )
      {
        v30 = 278;
        if ( v76 != 13 )
        {
          v30 = 279;
          if ( v76 != 14 )
          {
            v30 = 280;
            if ( v76 != 15 )
            {
              v30 = 729;
              if ( v76 == 16 )
                v30 = 281;
            }
          }
        }
      }
      goto LABEL_83;
    case 0x11D:
      v30 = 282;
      v75 = **(_WORD **)(a2 + 48);
      if ( v75 != 12 )
      {
        v30 = 283;
        if ( v75 != 13 )
        {
          v30 = 284;
          if ( v75 != 14 )
          {
            v30 = 285;
            if ( v75 != 15 )
            {
              v30 = 729;
              if ( v75 == 16 )
                v30 = 286;
            }
          }
        }
      }
      goto LABEL_83;
    case 0x11E:
      v30 = 287;
      v74 = **(_WORD **)(a2 + 48);
      if ( v74 != 12 )
      {
        v30 = 288;
        if ( v74 != 13 )
        {
          v30 = 289;
          if ( v74 != 14 )
          {
            v30 = 290;
            if ( v74 != 15 )
            {
              v30 = 729;
              if ( v74 == 16 )
                v30 = 291;
            }
          }
        }
      }
LABEL_83:
      result = sub_38062F0((__int64 *)a1, a2, v30);
      v11 = result;
      v13 = v32;
      goto LABEL_6;
    case 0x11F:
      HIDWORD(v84) = 0;
      v72 = sub_2FE5F00(**(_WORD **)(a2 + 48), *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL));
      goto LABEL_237;
    case 0x121:
      v84 = 0x100000000LL;
      v72 = sub_2FE5F60(**(_WORD **)(a2 + 48), *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL));
LABEL_237:
      result = sub_3808340((__int64 *)a1, a2, v72, v84);
      v11 = result;
      v13 = v73;
      goto LABEL_6;
    case 0x12A:
      result = (__int64)sub_37FCF70((__int64 *)a1, a2, a7);
      v11 = result;
      v13 = v71;
      goto LABEL_6;
    case 0x13D:
      result = (__int64)sub_37FD4B0((__int64 *)a1, a2);
      v11 = result;
      v13 = v70;
      goto LABEL_6;
    case 0x14F:
      result = (__int64)sub_3806750((__int64 *)a1, a2, a7);
      v11 = result;
      v13 = v69;
      goto LABEL_6;
    case 0x152:
      result = (__int64)sub_37FD230((__int64 *)a1, a2);
      v11 = result;
      v13 = v68;
      goto LABEL_6;
    case 0x156:
      result = (__int64)sub_3805050((__int64 *)a1, a2, a7);
      v11 = result;
      v13 = v65;
LABEL_6:
      if ( v11 )
        return sub_375F330(a1, a2, v7, v11, v13);
      return result;
    case 0x176:
    case 0x177:
      v8 = sub_346AFF0(a7, *(_QWORD *)a1, a2, *(_QWORD **)(a1 + 8), a4, a5);
      return sub_3760E70(a1, a2, 0, (unsigned __int64)v8, v9);
    case 0x178:
    case 0x179:
    case 0x17A:
    case 0x17B:
    case 0x17C:
    case 0x17D:
      v8 = sub_346A7D0(*(_QWORD *)a1, a2, *(_QWORD **)(a1 + 8), a7);
      return sub_3760E70(a1, a2, 0, (unsigned __int64)v8, v9);
    default:
      sub_C64ED0("Do not know how to soften the result of this operator!", 1u);
  }
}
