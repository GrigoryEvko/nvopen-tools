// Function: sub_21D7110
// Address: 0x21d7110
//
__int64 *__fastcall sub_21D7110(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 *a4,
        __int64 a5,
        int a6,
        __m128i a7,
        double a8,
        __m128i a9)
{
  __int64 *result; // rax
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // rax
  __int64 v16; // rax
  _QWORD *v17; // rcx
  __int64 v18; // rdx
  _QWORD *v19; // rax

  result = sub_2178FB0(a2, a3, a4, *(unsigned __int8 *)(a1[10193] + 936), a5, a6, a7, a8, a9);
  if ( !result )
  {
    v15 = *(unsigned __int16 *)(a2 + 24);
    if ( (__int16)v15 > 225 )
      return (__int64 *)a2;
    if ( (__int16)v15 <= 100 )
    {
      switch ( (__int16)v15 )
      {
        case 12:
          return (__int64 *)sub_21CF630(*(double *)a7.m128i_i64, a8, *(double *)a9.m128i_i64, (__int64)a1, a2, a3, a4);
        case 13:
        case 14:
        case 15:
        case 16:
        case 17:
        case 18:
        case 19:
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
        case 45:
        case 47:
        case 48:
        case 50:
          return (__int64 *)sub_21761F0(a2, a3, a4, v12, v13, v14);
        case 20:
        case 21:
          return 0;
        case 43:
          v16 = *(_QWORD *)(**(_QWORD **)(a2 + 32) + 88LL);
          v17 = *(_QWORD **)(v16 + 24);
          if ( *(_DWORD *)(v16 + 32) > 0x40u )
            v17 = (_QWORD *)*v17;
          result = (__int64 *)a2;
          if ( v17 == (_QWORD *)4004 )
            return (__int64 *)sub_2172010(a2, *(double *)a7.m128i_i64, a8, *(double *)a9.m128i_i64, a3, a4);
          return result;
        case 44:
          v18 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL) + 88LL);
          v19 = *(_QWORD **)(v18 + 24);
          if ( *(_DWORD *)(v18 + 32) > 0x40u )
            v19 = (_QWORD *)*v19;
          if ( v19 == (_QWORD *)4057 )
            return (__int64 *)sub_2173940(a2, a4, a7, a8, a9);
          if ( (unsigned __int64)v19 > 0xFD9 )
          {
            if ( v19 != (_QWORD *)4492 )
              return (__int64 *)a2;
          }
          else
          {
            if ( v19 == (_QWORD *)4043 )
              return (__int64 *)sub_2173940(a2, a4, a7, a8, a9);
            if ( v19 != (_QWORD *)4047 )
              return (__int64 *)a2;
          }
          return (__int64 *)sub_2173C80(a2, a4, a7, a8, a9);
        case 46:
          return sub_21D6D20(a7, a8, a9, (__int64)a1, a2, a3, a4);
        case 49:
          return (__int64 *)sub_21762A0(a2, a3, a4);
      }
    }
    switch ( (__int16)v15 )
    {
      case 101:
        result = sub_21D5570(*(double *)a7.m128i_i64, a8, a9, (__int64)a1, a2, a3, a4);
        break;
      case 102:
      case 103:
      case 105:
      case 108:
      case 110:
      case 111:
      case 112:
      case 113:
      case 114:
      case 115:
      case 116:
      case 117:
      case 118:
      case 119:
      case 120:
      case 121:
      case 122:
      case 123:
      case 124:
      case 125:
      case 126:
      case 127:
      case 128:
      case 129:
      case 130:
      case 131:
      case 132:
      case 133:
      case 135:
      case 136:
      case 138:
      case 142:
      case 143:
      case 144:
      case 145:
      case 146:
      case 147:
      case 148:
      case 149:
      case 150:
      case 151:
      case 152:
      case 153:
      case 154:
      case 155:
      case 156:
      case 157:
      case 158:
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
      case 208:
      case 209:
      case 210:
      case 211:
      case 212:
      case 213:
      case 214:
      case 215:
      case 216:
      case 217:
      case 218:
      case 219:
      case 220:
      case 221:
      case 222:
      case 223:
      case 224:
      case 225:
        result = sub_2176390(a2, a7, a8, a9, a3, a4);
        break;
      case 104:
        result = (__int64 *)sub_21D4F30(a7, a8, a9, (__int64)a1, a2, a3, a4);
        break;
      case 106:
        result = sub_21D5280(a7, a8, a9, (__int64)a1, a2, a3, a4);
        break;
      case 107:
        result = sub_21D4BD0(a7, a8, a9, (__int64)a1, a2, a3, a4);
        break;
      case 109:
        return (__int64 *)a2;
      case 134:
        result = (__int64 *)sub_21D5EE0(*(double *)a7.m128i_i64, a8, a9, (__int64)a1, a2, a3, a4);
        break;
      case 137:
        result = (__int64 *)sub_2176C60(a2, a3, a4, a7, a8, a9);
        break;
      case 139:
        result = sub_21D5AE0((__int64)a1, a2, a3, a4, *(double *)a7.m128i_i64, a8, a9);
        break;
      case 140:
      case 141:
        result = sub_21D56C0((__int64)a1, a2, a3, a4, *(double *)a7.m128i_i64, a8, a9);
        break;
      case 185:
        result = sub_21D6180((__int64)a1, a2, a3, a4, *(double *)a7.m128i_i64, a8, a9);
        break;
      case 186:
        result = (__int64 *)sub_21D6C00(a1, a2, a3, a4, *(double *)a7.m128i_i64, a8, a9);
        break;
      case 187:
        result = sub_21779A0(a2, *(double *)a7.m128i_i64, a8, a9, a3, a4, *(_BYTE *)(a1[10193] + 936));
        break;
      case 204:
        result = sub_21748C0(a2, *(double *)a7.m128i_i64, a8, a9, a3, a4);
        break;
      case 207:
        result = (__int64 *)sub_2175890(a2, a3, a4);
        break;
    }
  }
  return result;
}
