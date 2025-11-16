// Function: sub_2141F70
// Address: 0x2141f70
//
void __fastcall sub_2141F70(__int64 *a1, unsigned __int64 a2, unsigned int a3, __m128i a4, __m128i a5, __m128i a6)
{
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int16 v12; // si
  int v13; // eax
  const __m128i *v14; // r9
  const __m128i *v15; // r9
  __int64 v16; // rax
  __int64 v17; // rsi
  _QWORD *v18; // r10
  __int64 v19; // r14
  __int64 v20; // rdx
  __int64 v21; // r15
  __int64 v22; // r9
  __int128 *v23; // r11
  __int64 v24; // r8
  unsigned __int8 v25; // cl
  __int64 *v26; // r14
  __int16 *v27; // rdx
  __int16 *v28; // r15
  __int64 v29; // rsi
  __int64 *v30; // r10
  __int64 v31; // r8
  const void **v32; // r11
  unsigned int v33; // ecx
  __m128i *v34; // rdx
  const __m128i *v35; // r9
  const __m128i *v36; // r9
  unsigned __int8 v37; // [rsp+8h] [rbp-A0h]
  __int64 v38; // [rsp+10h] [rbp-98h]
  unsigned int v39; // [rsp+10h] [rbp-98h]
  __int64 v40; // [rsp+18h] [rbp-90h]
  __int64 v41; // [rsp+18h] [rbp-90h]
  __int128 *v42; // [rsp+20h] [rbp-88h]
  const void **v43; // [rsp+20h] [rbp-88h]
  __m128i *v44; // [rsp+20h] [rbp-88h]
  _QWORD *v45; // [rsp+28h] [rbp-80h]
  __int64 *v46; // [rsp+28h] [rbp-80h]
  __int64 *v47; // [rsp+28h] [rbp-80h]
  __int16 *v48; // [rsp+38h] [rbp-70h] BYREF
  __m128i *v49; // [rsp+40h] [rbp-68h]
  __m128i v50; // [rsp+48h] [rbp-60h] BYREF
  __int64 v51; // [rsp+58h] [rbp-50h] BYREF
  unsigned __int64 v52; // [rsp+60h] [rbp-48h]
  __int64 v53; // [rsp+68h] [rbp-40h]
  __m128i *v54; // [rsp+70h] [rbp-38h]

  v7 = a3;
  v8 = *(_QWORD *)(a2 + 40) + 16LL * a3;
  v50.m128i_i32[2] = 0;
  v48 = 0;
  LODWORD(v49) = 0;
  v9 = *(_QWORD *)(v8 + 8);
  v50.m128i_i64[0] = 0;
  if ( !(unsigned __int8)sub_2016240(a1, a2, *(_BYTE *)v8, v9, 1u, 0, 0) )
  {
    v12 = *(_WORD *)(a2 + 24);
    switch ( v12 )
    {
      case 3:
        sub_212EE50((__int64)a1, a2, (__int64)&v48, (__int64)&v50, a4, *(double *)a5.m128i_i64, a6);
        break;
      case 4:
        sub_212F1B0((__int64)a1, a2, (__int64)&v48, (__int64)&v50, a4, *(double *)a5.m128i_i64, a6);
        break;
      case 5:
      case 6:
      case 7:
      case 8:
      case 9:
      case 11:
      case 12:
      case 13:
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
      case 24:
      case 25:
      case 26:
      case 27:
      case 28:
      case 29:
      case 30:
      case 31:
      case 32:
      case 33:
      case 34:
      case 35:
      case 36:
      case 37:
      case 38:
      case 39:
      case 40:
      case 41:
      case 42:
      case 43:
      case 44:
      case 45:
      case 46:
      case 47:
      case 59:
      case 60:
      case 61:
      case 62:
      case 63:
      case 76:
      case 77:
      case 78:
      case 79:
      case 80:
      case 81:
      case 82:
      case 83:
      case 84:
      case 85:
      case 86:
      case 87:
      case 88:
      case 89:
      case 90:
      case 91:
      case 92:
      case 93:
      case 94:
      case 95:
      case 96:
      case 97:
      case 98:
      case 99:
      case 100:
      case 101:
      case 102:
      case 103:
      case 104:
      case 105:
      case 107:
      case 108:
      case 109:
      case 110:
      case 111:
      case 112:
      case 113:
      case 121:
      case 125:
      case 126:
      case 135:
      case 137:
      case 138:
      case 139:
      case 140:
      case 141:
      case 146:
      case 147:
      case 149:
      case 150:
      case 151:
      case 154:
      case 156:
      case 157:
      case 159:
      case 160:
      case 161:
      case 162:
      case 163:
      case 164:
      case 165:
      case 166:
      case 167:
      case 168:
      case 169:
      case 170:
      case 171:
      case 172:
      case 173:
      case 174:
      case 175:
      case 176:
      case 177:
      case 178:
      case 179:
      case 180:
      case 181:
      case 182:
      case 183:
      case 184:
      case 186:
      case 187:
      case 188:
      case 189:
      case 190:
      case 191:
      case 192:
      case 193:
      case 194:
      case 195:
      case 196:
      case 197:
      case 198:
      case 199:
      case 200:
      case 201:
      case 202:
      case 203:
      case 205:
      case 206:
      case 207:
      case 208:
      case 209:
      case 210:
      case 212:
      case 213:
      case 214:
      case 215:
      case 216:
      case 217:
      case 218:
      case 220:
      case 221:
      case 223:
      case 224:
      case 225:
      case 226:
      case 227:
      case 228:
      case 229:
      case 230:
      case 231:
      case 232:
      case 233:
      case 234:
        v13 = sub_1F403C0(v12, *(_BYTE *)(a2 + 88));
        sub_200E9A0((__int64)&v51, a1, v13, a2, 0);
        sub_200E870((__int64)a1, v51, v52, (__int64)&v48, &v50, a4, *(double *)a5.m128i_i64, a6);
        sub_2013400((__int64)a1, a2, 1, v53, v54, v14);
        break;
      case 10:
        sub_212F6A0(a1, a2, (__int64)&v48, (__int64)&v50, a4, *(double *)a5.m128i_i64, a6);
        break;
      case 48:
        sub_2147AE0(a1, a2, &v48, &v50);
        break;
      case 49:
        sub_2143C70(a1, a2, &v48, &v50);
        break;
      case 50:
        sub_2143C40(a1, a2, &v48, &v50);
        break;
      case 51:
        sub_2146BB0(a1, a2, (unsigned int)v7, &v48, &v50);
        break;
      case 52:
      case 53:
        sub_212CE20(a1, a2, (__int64)&v48, (__int64)&v50, v10, v11);
        break;
      case 54:
        sub_2131720((__int64)a1, a2, (__int64)&v48, (__int64)&v50, a4, *(double *)a5.m128i_i64, a6);
        break;
      case 55:
        sub_2132690((__m128i **)a1, a2, (__int64)&v48, &v50, *(double *)a4.m128i_i64, *(double *)a5.m128i_i64, a6);
        break;
      case 56:
        sub_21336F0((__m128i **)a1, a2, (__int64)&v48, &v50, *(double *)a4.m128i_i64, *(double *)a5.m128i_i64, a6);
        break;
      case 57:
        sub_21332C0((__m128i **)a1, a2, (__int64)&v48, &v50, *(double *)a4.m128i_i64, *(double *)a5.m128i_i64, a6);
        break;
      case 58:
        sub_21338E0((__m128i **)a1, a2, (__int64)&v48, &v50, *(double *)a4.m128i_i64, *(double *)a5.m128i_i64, a6);
        break;
      case 64:
      case 65:
        sub_212E1F0((__int64)a1, a2, (__int64)&v48, (__int64)&v50);
        break;
      case 66:
      case 67:
        sub_212E4D0((__int64)a1, a2, (__int64)&v48, (__int64)&v50);
        break;
      case 68:
      case 69:
        sub_212EBF0((__int64)a1, a2, (__int64)&v48, (__int64)&v50);
        break;
      case 70:
      case 72:
        sub_21322E0((__int64)a1, a2, (__int64)&v48, &v50, *(double *)a4.m128i_i64, *(double *)a5.m128i_i64, a6);
        break;
      case 71:
      case 73:
        sub_212E720(
          (__int64)a1,
          a2,
          (__int64)&v48,
          (__int64)&v50,
          *(double *)a4.m128i_i64,
          *(double *)a5.m128i_i64,
          a6,
          v10,
          v11);
        break;
      case 74:
      case 75:
        sub_2137A80((__m128i **)a1, a2, (__int64)&v48, &v50, a4, a5, a6);
        break;
      case 106:
        sub_2143D60(a1, a2, &v48, &v50);
        break;
      case 114:
      case 115:
      case 116:
      case 117:
        sub_212C800(a1, a2, (__int64)&v48, (__int64)&v50, *(double *)a4.m128i_i64, *(double *)a5.m128i_i64, a6);
        break;
      case 118:
      case 119:
      case 120:
        sub_21315A0((__int64)a1, a2, (__int64)&v48, (__int64)&v50, *(double *)a4.m128i_i64, *(double *)a5.m128i_i64, a6);
        break;
      case 122:
      case 123:
      case 124:
        sub_2132880((__int64)a1, a2, (__int64)&v48, &v50, a4, a5, a6);
        break;
      case 127:
        sub_212F590(
          (__int64)a1,
          a2,
          &v48,
          &v50,
          *(double *)a4.m128i_i64,
          *(double *)a5.m128i_i64,
          *(double *)a6.m128i_i64);
        break;
      case 128:
      case 132:
        sub_212FEB0((__int64 **)a1, a2, &v48, &v50, a4, *(double *)a5.m128i_i64, a6);
        break;
      case 129:
      case 133:
        sub_212F970((__int64 **)a1, a2, &v48, (__int64)&v50, a4, *(double *)a5.m128i_i64, a6);
        break;
      case 130:
        sub_212FD50((__int64)a1, a2, &v48, &v50, a4, *(double *)a5.m128i_i64, a6);
        break;
      case 131:
        sub_212F480(
          (__int64)a1,
          a2,
          &v48,
          &v50,
          *(double *)a4.m128i_i64,
          *(double *)a5.m128i_i64,
          *(double *)a6.m128i_i64);
        break;
      case 134:
        sub_2146C90(a1, a2, &v48, &v50);
        break;
      case 136:
        sub_2147770(a1, a2, &v48, &v50);
        break;
      case 142:
        sub_213DA80((__int64)a1, a2, (__int64)&v48, v50.m128i_i64, *(double *)a4.m128i_i64, *(double *)a5.m128i_i64, a6);
        break;
      case 143:
        sub_213DF50((__int64)a1, a2, (__int64)&v48, &v50, *(double *)a4.m128i_i64, *(double *)a5.m128i_i64, a6);
        break;
      case 144:
        sub_213D870((__int64)a1, a2, (__int64)&v48, (__int64)&v50, a4, *(double *)a5.m128i_i64, a6);
        break;
      case 145:
        sub_21334C0((__int64)a1, a2, (__int64)&v48, (__int64)&v50, a4, *(double *)a5.m128i_i64, a6);
        break;
      case 148:
        sub_2132E30((__int64)a1, a2, (__int64 *)&v48, (__int64)&v50, a4, *(double *)a5.m128i_i64, a6);
        break;
      case 152:
        sub_2141CB0((__int64)a1, a2, (__int64)&v48, &v50, *(double *)a4.m128i_i64, a5, a6);
        break;
      case 153:
        sub_2141E10((__int64)a1, a2, (__int64)&v48, &v50, *(double *)a4.m128i_i64, a5, a6);
        break;
      case 155:
        sub_2130290(a1, a2, (__int64)&v48, (__int64)&v50, a4, *(double *)a5.m128i_i64, a6);
        break;
      case 158:
        sub_2147DE0(a1, a2, &v48, &v50);
        break;
      case 185:
        sub_2130420((__int64)a1, a2, (__int64)&v48, (__int64)&v50);
        break;
      case 204:
        sub_2144790(a1, a2, &v48, &v50);
        break;
      case 211:
        sub_21321E0(a1, a2, (__int64)&v48, (__int64)&v50, *(double *)a4.m128i_i64, *(double *)a5.m128i_i64, a6);
        break;
      case 219:
        sub_2133AE0((__int64)a1, a2, a4, *(double *)a5.m128i_i64, a6, (__int64)&v48, (__int64)&v50, v10, v11);
        break;
      case 222:
        v16 = sub_1D252B0(a1[1], **(unsigned __int8 **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL), 1, 0);
        v17 = *(_QWORD *)(a2 + 72);
        v18 = (_QWORD *)a1[1];
        v19 = v16;
        v21 = v20;
        v22 = *(_QWORD *)(a2 + 104);
        v23 = *(__int128 **)(a2 + 32);
        v51 = v17;
        v24 = *(_QWORD *)(a2 + 96);
        v25 = *(_BYTE *)(a2 + 88);
        if ( v17 )
        {
          v37 = *(_BYTE *)(a2 + 88);
          v38 = *(_QWORD *)(a2 + 96);
          v40 = v22;
          v42 = v23;
          v45 = v18;
          sub_1623A60((__int64)&v51, v17, 2);
          v25 = v37;
          v18 = v45;
          v24 = v38;
          v22 = v40;
          v23 = v42;
        }
        LODWORD(v52) = *(_DWORD *)(a2 + 64);
        v26 = sub_1D24690(
                v18,
                0xDDu,
                (__int64)&v51,
                v25,
                v24,
                v22,
                v19,
                v21,
                *v23,
                *(__int128 *)((char *)v23 + 40),
                v23[5],
                *(__int128 *)((char *)v23 + 120));
        v28 = v27;
        if ( v51 )
          sub_161E7C0((__int64)&v51, v51);
        v29 = *(_QWORD *)(a2 + 72);
        v30 = (__int64 *)a1[1];
        v31 = *(_QWORD *)(a2 + 32);
        v32 = *(const void ***)(*(_QWORD *)(a2 + 40) + 24LL);
        v33 = *(unsigned __int8 *)(*(_QWORD *)(a2 + 40) + 16LL);
        v51 = v29;
        if ( v29 )
        {
          v39 = v33;
          v41 = v31;
          v43 = v32;
          v46 = v30;
          sub_1623A60((__int64)&v51, v29, 2);
          v33 = v39;
          v31 = v41;
          v32 = v43;
          v30 = v46;
        }
        LODWORD(v52) = *(_DWORD *)(a2 + 64);
        v47 = sub_1F81070(
                v30,
                (__int64)&v51,
                v33,
                v32,
                (unsigned __int64)v26,
                v28,
                (__m128)a4,
                *(double *)a5.m128i_i64,
                a6,
                *(_OWORD *)(v31 + 80),
                0x11u);
        v44 = v34;
        if ( v51 )
          sub_161E7C0((__int64)&v51, v51);
        sub_200E870(
          (__int64)a1,
          (__int64)v26,
          (unsigned __int64)v28,
          (__int64)&v48,
          &v50,
          a4,
          *(double *)a5.m128i_i64,
          a6);
        sub_2013400((__int64)a1, a2, 1, (__int64)v47, v44, v35);
        sub_2013400((__int64)a1, a2, 2, (__int64)v26, (__m128i *)1, v36);
        break;
    }
    if ( v48 )
      sub_2015C40((__int64)a1, a2, v7, (__int64)v48, v49, v15, v50.m128i_u64[0], v50.m128i_i64[1]);
  }
}
