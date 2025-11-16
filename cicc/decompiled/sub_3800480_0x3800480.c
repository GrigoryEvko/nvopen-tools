// Function: sub_3800480
// Address: 0x3800480
//
void __fastcall sub_3800480(__int64 *a1, unsigned __int64 a2, unsigned int a3, __m128i a4)
{
  __int64 v4; // rbx
  __int64 v5; // rax
  __int64 v6; // rcx
  __int64 v7; // r9
  int v8; // edx
  __int16 v9; // ax
  __int16 v10; // ax
  __int16 v11; // ax
  __int16 v12; // ax
  int v13; // edx
  __int16 v14; // ax
  __int16 v15; // ax
  __int16 v16; // ax
  __int16 v17; // ax
  __int16 v18; // ax
  __int16 v19; // ax
  __int16 v20; // ax
  __int16 v21; // ax
  __int16 v22; // ax
  __int16 v23; // ax
  __int16 v24; // ax
  __int16 v25; // ax
  __int16 v26; // ax
  __int16 v27; // ax
  __int16 v28; // ax
  __int16 v29; // ax
  __int16 v30; // ax
  __int16 v31; // ax
  __int16 v32; // ax
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
  int v46; // eax
  __int64 v47; // r8
  __int64 v48; // r9
  __int64 v49; // [rsp+8h] [rbp-48h]
  __m128i v50; // [rsp+10h] [rbp-40h] BYREF
  __m128i v51[3]; // [rsp+20h] [rbp-30h] BYREF

  v4 = a3;
  v5 = *(_QWORD *)(a2 + 48) + 16LL * a3;
  v51[0].m128i_i32[2] = 0;
  v50.m128i_i64[0] = 0;
  v50.m128i_i32[2] = 0;
  v6 = *(_QWORD *)(v5 + 8);
  v51[0].m128i_i64[0] = 0;
  if ( !(unsigned __int8)sub_3761870(a1, a2, *(_WORD *)v5, v6, 1) )
  {
    switch ( *(_DWORD *)(a2 + 24) )
    {
      case 0xC:
        sub_37FE110(a1, a2, (__int64)&v50, (__int64)v51, a4);
        break;
      case 0x33:
        sub_384A050(a1, a2, &v50, v51);
        break;
      case 0x34:
        sub_37FF2F0((__int64)a1, a2, (unsigned int *)&v50, (unsigned int *)v51, a4);
        break;
      case 0x35:
        sub_3846070(a1, a2, &v50, v51);
        break;
      case 0x36:
        sub_3846040(a1, a2, &v50, v51);
        break;
      case 0x37:
        sub_3845F80(a1, a2, (unsigned int)v4, &v50, v51);
        break;
      case 0x60:
      case 0x65:
        v8 = 55;
        v9 = **(_WORD **)(a2 + 48);
        if ( v9 != 12 )
        {
          v8 = 56;
          if ( v9 != 13 )
          {
            v8 = 57;
            if ( v9 != 14 )
            {
              v8 = 58;
              if ( v9 != 15 )
              {
                v8 = 729;
                if ( v9 == 16 )
                  v8 = 59;
              }
            }
          }
        }
        goto LABEL_15;
      case 0x61:
      case 0x66:
        v8 = 60;
        v10 = **(_WORD **)(a2 + 48);
        if ( v10 != 12 )
        {
          v8 = 61;
          if ( v10 != 13 )
          {
            v8 = 62;
            if ( v10 != 14 )
            {
              v8 = 63;
              if ( v10 != 15 )
              {
                v8 = 729;
                if ( v10 == 16 )
                  v8 = 64;
              }
            }
          }
        }
        goto LABEL_15;
      case 0x62:
      case 0x67:
        v8 = 65;
        v11 = **(_WORD **)(a2 + 48);
        if ( v11 != 12 )
        {
          v8 = 66;
          if ( v11 != 13 )
          {
            v8 = 67;
            if ( v11 != 14 )
            {
              v8 = 68;
              if ( v11 != 15 )
              {
                v8 = 729;
                if ( v11 == 16 )
                  v8 = 69;
              }
            }
          }
        }
        goto LABEL_15;
      case 0x63:
      case 0x68:
        v8 = 70;
        v12 = **(_WORD **)(a2 + 48);
        if ( v12 != 12 )
        {
          v8 = 71;
          if ( v12 != 13 )
          {
            v8 = 72;
            if ( v12 != 14 )
            {
              v8 = 73;
              if ( v12 != 15 )
              {
                v8 = 729;
                if ( v12 == 16 )
                  v8 = 74;
              }
            }
          }
        }
        goto LABEL_15;
      case 0x64:
      case 0x69:
        v8 = 75;
        v28 = **(_WORD **)(a2 + 48);
        if ( v28 != 12 )
        {
          v8 = 76;
          if ( v28 != 13 )
          {
            v8 = 77;
            if ( v28 != 14 )
            {
              v8 = 78;
              if ( v28 != 15 )
              {
                v8 = 729;
                if ( v28 == 16 )
                  v8 = 79;
              }
            }
          }
        }
        goto LABEL_15;
      case 0x6A:
      case 0x96:
        sub_37FEB80(a1, a2, (__int64)&v50, (__int64)v51, a4);
        break;
      case 0x6B:
      case 0xF6:
        v13 = 90;
        v29 = **(_WORD **)(a2 + 48);
        if ( v29 != 12 )
        {
          v13 = 91;
          if ( v29 != 13 )
          {
            v13 = 92;
            if ( v29 != 14 )
            {
              v13 = 93;
              if ( v29 != 15 )
              {
                v13 = 729;
                if ( v29 == 16 )
                  v13 = 94;
              }
            }
          }
        }
        goto LABEL_43;
      case 0x6C:
      case 0x101:
        v8 = 212;
        v30 = **(_WORD **)(a2 + 48);
        if ( v30 != 12 )
        {
          v8 = 213;
          if ( v30 != 13 )
          {
            v8 = 214;
            if ( v30 != 14 )
            {
              v8 = 215;
              if ( v30 != 15 )
              {
                v8 = 729;
                if ( v30 == 16 )
                  v8 = 216;
              }
            }
          }
        }
        goto LABEL_15;
      case 0x6D:
      case 0x102:
        v8 = sub_2FE5E70(**(_WORD **)(a2 + 48), *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL));
        goto LABEL_15;
      case 0x6E:
      case 0x103:
        v8 = sub_2FE5EA0(**(_WORD **)(a2 + 48), *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL));
        goto LABEL_15;
      case 0x6F:
      case 0xF8:
        v13 = 155;
        v31 = **(_WORD **)(a2 + 48);
        if ( v31 != 12 )
        {
          v13 = 156;
          if ( v31 != 13 )
          {
            v13 = 157;
            if ( v31 != 14 )
            {
              v13 = 158;
              if ( v31 != 15 )
              {
                v13 = 729;
                if ( v31 == 16 )
                  v13 = 159;
              }
            }
          }
        }
        goto LABEL_43;
      case 0x70:
      case 0xF9:
        v13 = 160;
        v32 = **(_WORD **)(a2 + 48);
        if ( v32 != 12 )
        {
          v13 = 161;
          if ( v32 != 13 )
          {
            v13 = 162;
            if ( v32 != 14 )
            {
              v13 = 163;
              if ( v32 != 15 )
              {
                v13 = 729;
                if ( v32 == 16 )
                  v13 = 164;
              }
            }
          }
        }
        goto LABEL_43;
      case 0x71:
      case 0xFA:
        v13 = 165;
        v33 = **(_WORD **)(a2 + 48);
        if ( v33 != 12 )
        {
          v13 = 166;
          if ( v33 != 13 )
          {
            v13 = 167;
            if ( v33 != 14 )
            {
              v13 = 168;
              if ( v33 != 15 )
              {
                v13 = 729;
                if ( v33 == 16 )
                  v13 = 169;
              }
            }
          }
        }
        goto LABEL_43;
      case 0x72:
      case 0xFB:
        v13 = 185;
        v34 = **(_WORD **)(a2 + 48);
        if ( v34 != 12 )
        {
          v13 = 186;
          if ( v34 != 13 )
          {
            v13 = 187;
            if ( v34 != 14 )
            {
              v13 = 188;
              if ( v34 != 15 )
              {
                v13 = 729;
                if ( v34 == 16 )
                  v13 = 189;
              }
            }
          }
        }
        goto LABEL_43;
      case 0x73:
      case 0xFC:
        v13 = 190;
        v35 = **(_WORD **)(a2 + 48);
        if ( v35 != 12 )
        {
          v13 = 191;
          if ( v35 != 13 )
          {
            v13 = 192;
            if ( v35 != 14 )
            {
              v13 = 193;
              if ( v35 != 15 )
              {
                v13 = 729;
                if ( v35 == 16 )
                  v13 = 194;
              }
            }
          }
        }
        goto LABEL_43;
      case 0x74:
      case 0xFD:
        v13 = 195;
        v36 = **(_WORD **)(a2 + 48);
        if ( v36 != 12 )
        {
          v13 = 196;
          if ( v36 != 13 )
          {
            v13 = 197;
            if ( v36 != 14 )
            {
              v13 = 198;
              if ( v36 != 15 )
              {
                v13 = 729;
                if ( v36 == 16 )
                  v13 = 199;
              }
            }
          }
        }
        goto LABEL_43;
      case 0x75:
      case 0x104:
        v8 = 200;
        v37 = **(_WORD **)(a2 + 48);
        if ( v37 != 12 )
        {
          v8 = 201;
          if ( v37 != 13 )
          {
            v8 = 202;
            if ( v37 != 14 )
            {
              v8 = 203;
              if ( v37 != 15 )
              {
                v8 = 729;
                if ( v37 == 16 )
                  v8 = 204;
              }
            }
          }
        }
        goto LABEL_15;
      case 0x76:
      case 0xFE:
        v13 = 170;
        v38 = **(_WORD **)(a2 + 48);
        if ( v38 != 12 )
        {
          v13 = 171;
          if ( v38 != 13 )
          {
            v13 = 172;
            if ( v38 != 14 )
            {
              v13 = 173;
              if ( v38 != 15 )
              {
                v13 = 729;
                if ( v38 == 16 )
                  v13 = 174;
              }
            }
          }
        }
        goto LABEL_43;
      case 0x77:
      case 0xFF:
        v13 = 175;
        v39 = **(_WORD **)(a2 + 48);
        if ( v39 != 12 )
        {
          v13 = 176;
          if ( v39 != 13 )
          {
            v13 = 177;
            if ( v39 != 14 )
            {
              v13 = 178;
              if ( v39 != 15 )
              {
                v13 = 729;
                if ( v39 == 16 )
                  v13 = 179;
              }
            }
          }
        }
        goto LABEL_43;
      case 0x78:
      case 0x100:
        v13 = 180;
        v40 = **(_WORD **)(a2 + 48);
        if ( v40 != 12 )
        {
          v13 = 181;
          if ( v40 != 13 )
          {
            v13 = 182;
            if ( v40 != 14 )
            {
              v13 = 183;
              if ( v40 != 15 )
              {
                v13 = 729;
                if ( v40 == 16 )
                  v13 = 184;
              }
            }
          }
        }
        goto LABEL_43;
      case 0x79:
      case 0x109:
        v13 = 130;
        v20 = **(_WORD **)(a2 + 48);
        if ( v20 != 12 )
        {
          v13 = 131;
          if ( v20 != 13 )
          {
            v13 = 132;
            if ( v20 != 14 )
            {
              v13 = 133;
              if ( v20 != 15 )
              {
                v13 = 729;
                if ( v20 == 16 )
                  v13 = 134;
              }
            }
          }
        }
        goto LABEL_43;
      case 0x7A:
      case 0x10A:
        v13 = 140;
        v21 = **(_WORD **)(a2 + 48);
        if ( v21 != 12 )
        {
          v13 = 141;
          if ( v21 != 13 )
          {
            v13 = 142;
            if ( v21 != 14 )
            {
              v13 = 143;
              if ( v21 != 15 )
              {
                v13 = 729;
                if ( v21 == 16 )
                  v13 = 144;
              }
            }
          }
        }
        goto LABEL_43;
      case 0x7B:
      case 0x106:
        v13 = 100;
        v22 = **(_WORD **)(a2 + 48);
        if ( v22 != 12 )
        {
          v13 = 101;
          if ( v22 != 13 )
          {
            v13 = 102;
            if ( v22 != 14 )
            {
              v13 = 103;
              if ( v22 != 15 )
              {
                v13 = 729;
                if ( v22 == 16 )
                  v13 = 104;
              }
            }
          }
        }
        goto LABEL_43;
      case 0x7C:
      case 0x108:
        v13 = 120;
        v23 = **(_WORD **)(a2 + 48);
        if ( v23 != 12 )
        {
          v13 = 121;
          if ( v23 != 13 )
          {
            v13 = 122;
            if ( v23 != 14 )
            {
              v13 = 123;
              if ( v23 != 15 )
              {
                v13 = 729;
                if ( v23 == 16 )
                  v13 = 124;
              }
            }
          }
        }
        goto LABEL_43;
      case 0x7D:
      case 0x107:
        v13 = 110;
        v24 = **(_WORD **)(a2 + 48);
        if ( v24 != 12 )
        {
          v13 = 111;
          if ( v24 != 13 )
          {
            v13 = 112;
            if ( v24 != 14 )
            {
              v13 = 113;
              if ( v24 != 15 )
              {
                v13 = 729;
                if ( v24 == 16 )
                  v13 = 114;
              }
            }
          }
        }
        goto LABEL_43;
      case 0x7E:
      case 0x10E:
        v13 = 232;
        v25 = **(_WORD **)(a2 + 48);
        if ( v25 != 12 )
        {
          v13 = 233;
          if ( v25 != 13 )
          {
            v13 = 234;
            if ( v25 != 14 )
            {
              v13 = 235;
              if ( v25 != 15 )
              {
                v13 = 729;
                if ( v25 == 16 )
                  v13 = 236;
              }
            }
          }
        }
        goto LABEL_43;
      case 0x7F:
      case 0x10F:
        v13 = 237;
        v26 = **(_WORD **)(a2 + 48);
        if ( v26 != 12 )
        {
          v13 = 238;
          if ( v26 != 13 )
          {
            v13 = 239;
            if ( v26 != 14 )
            {
              v13 = 240;
              if ( v26 != 15 )
              {
                v13 = 729;
                if ( v26 == 16 )
                  v13 = 241;
              }
            }
          }
        }
        goto LABEL_43;
      case 0x80:
      case 0x118:
        v8 = 267;
        v27 = **(_WORD **)(a2 + 48);
        if ( v27 != 12 )
        {
          v8 = 268;
          if ( v27 != 13 )
          {
            v8 = 269;
            if ( v27 != 14 )
            {
              v8 = 270;
              if ( v27 != 15 )
              {
                v8 = 729;
                if ( v27 == 16 )
                  v8 = 271;
              }
            }
          }
        }
        goto LABEL_15;
      case 0x81:
      case 0x117:
        v8 = 262;
        v16 = **(_WORD **)(a2 + 48);
        if ( v16 != 12 )
        {
          v8 = 263;
          if ( v16 != 13 )
          {
            v8 = 264;
            if ( v16 != 14 )
            {
              v8 = 265;
              if ( v16 != 15 )
              {
                v8 = 729;
                if ( v16 == 16 )
                  v8 = 266;
              }
            }
          }
        }
        goto LABEL_15;
      case 0x82:
      case 0x10C:
        v13 = 222;
        v17 = **(_WORD **)(a2 + 48);
        if ( v17 != 12 )
        {
          v13 = 223;
          if ( v17 != 13 )
          {
            v13 = 224;
            if ( v17 != 14 )
            {
              v13 = 225;
              if ( v17 != 15 )
              {
                v13 = 729;
                if ( v17 == 16 )
                  v13 = 226;
              }
            }
          }
        }
        goto LABEL_43;
      case 0x83:
      case 0x112:
        v13 = 252;
        v18 = **(_WORD **)(a2 + 48);
        if ( v18 != 12 )
        {
          v13 = 253;
          if ( v18 != 13 )
          {
            v13 = 254;
            if ( v18 != 14 )
            {
              v13 = 255;
              if ( v18 != 15 )
              {
                v13 = 729;
                if ( v18 == 16 )
                  v13 = 256;
              }
            }
          }
        }
        goto LABEL_43;
      case 0x84:
      case 0x110:
        v13 = 242;
        v19 = **(_WORD **)(a2 + 48);
        if ( v19 != 12 )
        {
          v13 = 243;
          if ( v19 != 13 )
          {
            v13 = 244;
            if ( v19 != 14 )
            {
              v13 = 245;
              if ( v19 != 15 )
              {
                v13 = 729;
                if ( v19 == 16 )
                  v13 = 246;
              }
            }
          }
        }
        goto LABEL_43;
      case 0x85:
      case 0x111:
        v13 = 247;
        v14 = **(_WORD **)(a2 + 48);
        if ( v14 != 12 )
        {
          v13 = 248;
          if ( v14 != 13 )
          {
            v13 = 249;
            if ( v14 != 14 )
            {
              v13 = 250;
              if ( v14 != 15 )
              {
                v13 = 729;
                if ( v14 == 16 )
                  v13 = 251;
              }
            }
          }
        }
        goto LABEL_43;
      case 0x86:
      case 0x10D:
        v13 = 227;
        v15 = **(_WORD **)(a2 + 48);
        if ( v15 != 12 )
        {
          v13 = 228;
          if ( v15 != 13 )
          {
            v13 = 229;
            if ( v15 != 14 )
            {
              v13 = 230;
              if ( v15 != 15 )
              {
                v13 = 729;
                if ( v15 == 16 )
                  v13 = 231;
              }
            }
          }
        }
        goto LABEL_43;
      case 0x8F:
      case 0x90:
      case 0xDC:
      case 0xDD:
        sub_37FF700(a1, a2, &v50, v51);
        break;
      case 0x92:
      case 0xE9:
        sub_37FEF00(a1, a2, (__int64)&v50, (__int64)v51, a4);
        break;
      case 0x98:
        v8 = 257;
        v41 = **(_WORD **)(a2 + 48);
        if ( v41 != 12 )
        {
          v8 = 258;
          if ( v41 != 13 )
          {
            v8 = 259;
            if ( v41 != 14 )
            {
              v8 = 260;
              if ( v41 != 15 )
              {
                v8 = 729;
                if ( v41 == 16 )
                  v8 = 261;
              }
            }
          }
        }
        goto LABEL_15;
      case 0x9E:
        sub_3846160(a1, a2, &v50, v51);
        break;
      case 0xCD:
        sub_3849200(a1, a2, &v50, v51);
        break;
      case 0xCF:
        sub_3849C90(a1, a2, &v50, v51);
        break;
      case 0xEA:
        sub_384A7E0(a1, a2, &v50, v51);
        break;
      case 0xF4:
        sub_37FEDF0((__int64)a1, a2, (unsigned int *)&v50, (unsigned int *)v51, a4);
        break;
      case 0xF5:
        sub_37FE9C0((__int64)a1, a2, &v50, v51, a4);
        break;
      case 0xF7:
        v13 = 95;
        v42 = **(_WORD **)(a2 + 48);
        if ( v42 != 12 )
        {
          v13 = 96;
          if ( v42 != 13 )
          {
            v13 = 97;
            if ( v42 != 14 )
            {
              v13 = 98;
              if ( v42 != 15 )
              {
                v13 = 729;
                if ( v42 == 16 )
                  v13 = 99;
              }
            }
          }
        }
        goto LABEL_43;
      case 0x10B:
        v13 = 150;
        v43 = **(_WORD **)(a2 + 48);
        if ( v43 != 12 )
        {
          v13 = 151;
          if ( v43 != 13 )
          {
            v13 = 152;
            if ( v43 != 14 )
            {
              v13 = 153;
              if ( v43 != 15 )
              {
                v13 = 729;
                if ( v43 == 16 )
                  v13 = 154;
              }
            }
          }
        }
LABEL_43:
        sub_37FE4B0(a1, a2, v13, (__int64)&v50, (__int64)v51, a4);
        break;
      case 0x11D:
        v8 = 282;
        v44 = **(_WORD **)(a2 + 48);
        if ( v44 != 12 )
        {
          v8 = 283;
          if ( v44 != 13 )
          {
            v8 = 284;
            if ( v44 != 14 )
            {
              v8 = 285;
              if ( v44 != 15 )
              {
                v8 = 729;
                if ( v44 == 16 )
                  v8 = 286;
              }
            }
          }
        }
        goto LABEL_15;
      case 0x11E:
        v8 = 287;
        v45 = **(_WORD **)(a2 + 48);
        if ( v45 != 12 )
        {
          v8 = 288;
          if ( v45 != 13 )
          {
            v8 = 289;
            if ( v45 != 14 )
            {
              v8 = 290;
              if ( v45 != 15 )
              {
                v8 = 729;
                if ( v45 == 16 )
                  v8 = 291;
              }
            }
          }
        }
LABEL_15:
        sub_37FE690(a1, a2, v8, (__int64)&v50, (__int64)v51, a4);
        break;
      case 0x11F:
        HIDWORD(v49) = 0;
        v46 = sub_2FE5F00(**(_WORD **)(a2 + 48), *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL));
        goto LABEL_277;
      case 0x120:
        HIDWORD(v49) = 0;
        v46 = sub_2FE5F30(**(_WORD **)(a2 + 48), *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL));
        goto LABEL_277;
      case 0x121:
        v49 = 0x100000000LL;
        v46 = sub_2FE5F60(**(_WORD **)(a2 + 48), *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL));
LABEL_277:
        sub_37FE890(a1, a2, v46, v49, a4, v47, v48);
        break;
      case 0x12A:
        sub_37FF400((__int64)a1, a2, (__int64)&v50, (__int64)v51);
        break;
      case 0x13D:
        sub_3846C20(a1, a2, &v50, v51);
        break;
      default:
        sub_C64ED0("Do not know how to expand the result of this operator!", 1u);
    }
    if ( v50.m128i_i64[0] )
      sub_37604D0((__int64)a1, a2, v4, v50.m128i_u64[0], v50.m128i_i64[1], v7, v51[0].m128i_u64[0], v51[0].m128i_i64[1]);
  }
}
