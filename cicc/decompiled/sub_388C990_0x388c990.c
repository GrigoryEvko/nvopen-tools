// Function: sub_388C990
// Address: 0x388c990
//
__int64 __fastcall sub_388C990(__int64 a1, __m128i *a2)
{
  unsigned int v2; // r14d
  int v5; // eax
  unsigned int v6; // eax
  unsigned __int64 v8; // rsi
  __int64 v9; // rdi
  __int64 v10[2]; // [rsp+10h] [rbp-50h] BYREF
  char v11; // [rsp+20h] [rbp-40h]
  char v12; // [rsp+21h] [rbp-3Fh]

  v2 = 0;
  sub_1560670((__int64)a2);
  v5 = *(_DWORD *)(a1 + 64);
  while ( 2 )
  {
    switch ( v5 )
    {
      case 88:
        v6 = sub_388C5A0(a1, (unsigned int *)v10);
        if ( (_BYTE)v6 )
          return v6;
        sub_1560C00(a2, v10[0]);
        v5 = *(_DWORD *)(a1 + 64);
        continue;
      case 96:
      case 144:
      case 145:
      case 146:
      case 147:
      case 148:
      case 151:
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
        v12 = 1;
        v10[0] = (__int64)"invalid use of function-only attribute";
        goto LABEL_7;
      case 149:
      case 150:
      case 162:
      case 165:
      case 181:
      case 190:
      case 194:
      case 195:
        v12 = 1;
        v10[0] = (__int64)"invalid use of parameter-only attribute";
        goto LABEL_7;
      case 153:
        v6 = sub_388C650(a1, 153, v10);
        if ( (_BYTE)v6 )
          return v6;
        sub_1560C40(a2, v10[0]);
        v5 = *(_DWORD *)(a1 + 64);
        continue;
      case 154:
        v6 = sub_388C650(a1, 154, v10);
        if ( (_BYTE)v6 )
          return v6;
        sub_1560C60(a2, v10[0]);
        v5 = *(_DWORD *)(a1 + 64);
        continue;
      case 158:
        sub_15606E0(a2, 12);
        v9 = a1 + 8;
        goto LABEL_8;
      case 163:
        sub_15606E0(a2, 20);
        v9 = a1 + 8;
        goto LABEL_8;
      case 171:
        sub_15606E0(a2, 32);
        v9 = a1 + 8;
        goto LABEL_8;
      case 179:
      case 180:
        v12 = 1;
        v10[0] = (__int64)"invalid use of attribute on return type";
LABEL_7:
        v8 = *(_QWORD *)(a1 + 56);
        v11 = 3;
        v9 = a1 + 8;
        v2 |= sub_38814C0(a1 + 8, v8, (__int64)v10);
        goto LABEL_8;
      case 183:
        sub_15606E0(a2, 40);
        v9 = a1 + 8;
        goto LABEL_8;
      case 198:
        sub_15606E0(a2, 58);
        v9 = a1 + 8;
LABEL_8:
        v5 = sub_3887100(v9);
        *(_DWORD *)(a1 + 64) = v5;
        continue;
      case 377:
        v6 = sub_388BFE0(a1, a2);
        if ( !(_BYTE)v6 )
        {
          v5 = *(_DWORD *)(a1 + 64);
          continue;
        }
        return v6;
      default:
        return v2;
    }
  }
}
