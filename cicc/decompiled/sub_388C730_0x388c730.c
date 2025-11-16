// Function: sub_388C730
// Address: 0x388c730
//
__int64 __fastcall sub_388C730(__int64 a1, __m128i *a2)
{
  unsigned int v2; // r15d
  int v4; // eax
  unsigned int v5; // eax
  unsigned __int64 v7; // rsi
  __int64 v8[2]; // [rsp+10h] [rbp-50h] BYREF
  char v9; // [rsp+20h] [rbp-40h]
  char v10; // [rsp+21h] [rbp-3Fh]

  v2 = 0;
  sub_1560670((__int64)a2);
  v4 = *(_DWORD *)(a1 + 64);
  while ( 2 )
  {
    switch ( v4 )
    {
      case 88:
        v5 = sub_388C5A0(a1, (unsigned int *)v8);
        if ( (_BYTE)v5 )
          return v5;
        sub_1560C00(a2, v8[0]);
        v4 = *(_DWORD *)(a1 + 64);
        continue;
      case 96:
      case 144:
      case 145:
      case 146:
      case 147:
      case 148:
      case 157:
      case 159:
      case 160:
      case 161:
      case 164:
      case 166:
      case 167:
      case 168:
      case 170:
      case 172:
      case 173:
      case 174:
      case 175:
      case 176:
      case 177:
      case 178:
      case 182:
      case 185:
      case 186:
      case 187:
      case 188:
      case 189:
      case 191:
      case 192:
      case 193:
      case 196:
        v7 = *(_QWORD *)(a1 + 56);
        v10 = 1;
        v9 = 3;
        v8[0] = (__int64)"invalid use of function-only attribute";
        v2 |= sub_38814C0(a1 + 8, v7, (__int64)v8);
        goto LABEL_7;
      case 149:
        sub_15606E0(a2, 6);
        goto LABEL_7;
      case 150:
        sub_15606E0(a2, 11);
        goto LABEL_7;
      case 153:
        v5 = sub_388C650(a1, 153, v8);
        if ( (_BYTE)v5 )
          return v5;
        sub_1560C40(a2, v8[0]);
        v4 = *(_DWORD *)(a1 + 64);
        continue;
      case 154:
        v5 = sub_388C650(a1, 154, v8);
        if ( (_BYTE)v5 )
          return v5;
        sub_1560C60(a2, v8[0]);
        v4 = *(_DWORD *)(a1 + 64);
        continue;
      case 158:
        sub_15606E0(a2, 12);
        goto LABEL_7;
      case 162:
        sub_15606E0(a2, 19);
        goto LABEL_7;
      case 163:
        sub_15606E0(a2, 20);
        goto LABEL_7;
      case 165:
        sub_15606E0(a2, 22);
        goto LABEL_7;
      case 171:
        sub_15606E0(a2, 32);
        goto LABEL_7;
      case 179:
        sub_15606E0(a2, 36);
        goto LABEL_7;
      case 180:
        sub_15606E0(a2, 37);
        goto LABEL_7;
      case 181:
        sub_15606E0(a2, 38);
        goto LABEL_7;
      case 183:
        sub_15606E0(a2, 40);
        goto LABEL_7;
      case 190:
        sub_15606E0(a2, 53);
        goto LABEL_7;
      case 194:
        sub_15606E0(a2, 54);
        goto LABEL_7;
      case 195:
        sub_15606E0(a2, 55);
        goto LABEL_7;
      case 197:
        sub_15606E0(a2, 57);
        goto LABEL_7;
      case 198:
        sub_15606E0(a2, 58);
LABEL_7:
        v4 = sub_3887100(a1 + 8);
        *(_DWORD *)(a1 + 64) = v4;
        continue;
      case 377:
        v5 = sub_388BFE0(a1, a2);
        if ( !(_BYTE)v5 )
        {
          v4 = *(_DWORD *)(a1 + 64);
          continue;
        }
        return v5;
      default:
        return v2;
    }
  }
}
