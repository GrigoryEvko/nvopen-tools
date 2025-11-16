// Function: sub_388FCA0
// Address: 0x388fca0
//
__int64 __fastcall sub_388FCA0(__int64 a1, __m128i *a2, __int64 a3, char a4, _QWORD *a5)
{
  int v8; // eax
  __int64 v9; // r8
  __int64 result; // rax
  const char *v11; // rax
  unsigned __int64 v12; // rsi
  char v13; // al
  int v14; // eax
  _BYTE *v15; // rsi
  unsigned __int64 v16; // rsi
  __int64 v17; // [rsp+8h] [rbp-78h]
  unsigned __int8 v20; // [rsp+1Fh] [rbp-61h]
  unsigned int v21; // [rsp+2Ch] [rbp-54h] BYREF
  _QWORD v22[2]; // [rsp+30h] [rbp-50h] BYREF
  char v23; // [rsp+40h] [rbp-40h]
  char v24; // [rsp+41h] [rbp-3Fh]

  sub_1560670((__int64)a2);
  v20 = 0;
  v8 = *(_DWORD *)(a1 + 64);
  while ( 2 )
  {
    while ( v8 == 148 )
    {
      *a5 = *(_QWORD *)(a1 + 56);
      sub_15606E0(a2, 5);
      v9 = a1 + 8;
LABEL_5:
      v8 = sub_3887100(v9);
      *(_DWORD *)(a1 + 64) = v8;
    }
    switch ( v8 )
    {
      case 9:
        return 0;
      case 88:
        if ( a4 )
        {
          *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
          if ( (unsigned __int8)sub_388AF10(a1, 3, "expected '=' here") || (unsigned __int8)sub_388BA90(a1, v22) )
            return 1;
        }
        else if ( (unsigned __int8)sub_388C5A0(a1, (unsigned int *)v22) )
        {
          return 1;
        }
        sub_1560C00(a2, v22[0]);
        v8 = *(_DWORD *)(a1 + 64);
        continue;
      case 96:
        if ( !a4 )
        {
          if ( (unsigned __int8)sub_388D030(a1, v22) )
            return 1;
LABEL_71:
          sub_1560C20(a2, v22[0]);
          v8 = *(_DWORD *)(a1 + 64);
          continue;
        }
        *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
        if ( !(unsigned __int8)sub_388AF10(a1, 3, "expected '=' here") && !(unsigned __int8)sub_388BA90(a1, v22) )
          goto LABEL_71;
        return 1;
      case 143:
        BYTE4(v22[0]) = 0;
        result = sub_388CC90(a1, &v21, (__int64)v22);
        if ( (_BYTE)result )
          return result;
        sub_1560C90(a2, v21, (unsigned int *)v22);
        v8 = *(_DWORD *)(a1 + 64);
        continue;
      case 144:
        sub_15606E0(a2, 3);
        v9 = a1 + 8;
        goto LABEL_5;
      case 145:
        sub_15606E0(a2, 4);
        v9 = a1 + 8;
        goto LABEL_5;
      case 146:
        sub_15606E0(a2, 42);
        v9 = a1 + 8;
        goto LABEL_5;
      case 147:
        sub_15606E0(a2, 43);
        v9 = a1 + 8;
        goto LABEL_5;
      case 149:
      case 150:
      case 153:
      case 154:
      case 162:
      case 163:
      case 165:
      case 171:
      case 181:
      case 190:
      case 194:
      case 195:
        v24 = 1;
        v11 = "invalid use of parameter-only attribute on a function";
        goto LABEL_9;
      case 151:
        sub_15606E0(a2, 7);
        v9 = a1 + 8;
        goto LABEL_5;
      case 152:
        sub_15606E0(a2, 8);
        v9 = a1 + 8;
        goto LABEL_5;
      case 155:
        sub_15606E0(a2, 13);
        v9 = a1 + 8;
        goto LABEL_5;
      case 156:
        sub_15606E0(a2, 14);
        v9 = a1 + 8;
        goto LABEL_5;
      case 157:
        sub_15606E0(a2, 15);
        v9 = a1 + 8;
        goto LABEL_5;
      case 158:
      case 183:
      case 198:
        v24 = 1;
        v11 = "invalid use of attribute on a function";
LABEL_9:
        v22[0] = v11;
        v12 = *(_QWORD *)(a1 + 56);
        v9 = a1 + 8;
        v23 = 3;
        goto LABEL_10;
      case 159:
        sub_15606E0(a2, 16);
        v9 = a1 + 8;
        goto LABEL_5;
      case 160:
        sub_15606E0(a2, 17);
        v9 = a1 + 8;
        goto LABEL_5;
      case 161:
        sub_15606E0(a2, 18);
        v9 = a1 + 8;
        goto LABEL_5;
      case 164:
        sub_15606E0(a2, 21);
        v9 = a1 + 8;
        goto LABEL_5;
      case 166:
        sub_15606E0(a2, 24);
        v9 = a1 + 8;
        goto LABEL_5;
      case 167:
        sub_15606E0(a2, 25);
        v9 = a1 + 8;
        goto LABEL_5;
      case 168:
        sub_15606E0(a2, 26);
        v9 = a1 + 8;
        goto LABEL_5;
      case 169:
        sub_15606E0(a2, 27);
        v9 = a1 + 8;
        goto LABEL_5;
      case 170:
        sub_15606E0(a2, 31);
        v9 = a1 + 8;
        goto LABEL_5;
      case 172:
        sub_15606E0(a2, 28);
        v9 = a1 + 8;
        goto LABEL_5;
      case 173:
        sub_15606E0(a2, 29);
        v9 = a1 + 8;
        goto LABEL_5;
      case 174:
        sub_15606E0(a2, 23);
        v9 = a1 + 8;
        goto LABEL_5;
      case 175:
        sub_15606E0(a2, 30);
        v9 = a1 + 8;
        goto LABEL_5;
      case 176:
        sub_15606E0(a2, 33);
        v9 = a1 + 8;
        goto LABEL_5;
      case 177:
        sub_15606E0(a2, 35);
        v9 = a1 + 8;
        goto LABEL_5;
      case 178:
        sub_15606E0(a2, 34);
        v9 = a1 + 8;
        goto LABEL_5;
      case 179:
        sub_15606E0(a2, 36);
        v9 = a1 + 8;
        goto LABEL_5;
      case 180:
        sub_15606E0(a2, 37);
        v9 = a1 + 8;
        goto LABEL_5;
      case 182:
        sub_15606E0(a2, 39);
        v9 = a1 + 8;
        goto LABEL_5;
      case 184:
        sub_15606E0(a2, 47);
        v9 = a1 + 8;
        goto LABEL_5;
      case 185:
        sub_15606E0(a2, 49);
        v9 = a1 + 8;
        goto LABEL_5;
      case 186:
        sub_15606E0(a2, 50);
        v9 = a1 + 8;
        goto LABEL_5;
      case 187:
        sub_15606E0(a2, 51);
        v9 = a1 + 8;
        goto LABEL_5;
      case 188:
        sub_15606E0(a2, 41);
        v9 = a1 + 8;
        goto LABEL_5;
      case 189:
        sub_15606E0(a2, 46);
        v9 = a1 + 8;
        goto LABEL_5;
      case 191:
        sub_15606E0(a2, 45);
        v9 = a1 + 8;
        goto LABEL_5;
      case 192:
        sub_15606E0(a2, 44);
        v9 = a1 + 8;
        goto LABEL_5;
      case 193:
        sub_15606E0(a2, 52);
        v9 = a1 + 8;
        goto LABEL_5;
      case 196:
        sub_15606E0(a2, 56);
        v9 = a1 + 8;
        goto LABEL_5;
      case 197:
        sub_15606E0(a2, 57);
        v9 = a1 + 8;
        goto LABEL_5;
      case 370:
        v9 = a1 + 8;
        if ( a4 )
        {
          v24 = 1;
          v12 = *(_QWORD *)(a1 + 56);
          v22[0] = "cannot have an attribute group reference in an attribute group";
          v23 = 3;
LABEL_10:
          v17 = v9;
          v13 = sub_38814C0(v9, v12, (__int64)v22);
          v9 = v17;
          v20 |= v13;
        }
        else
        {
          v14 = *(_DWORD *)(a1 + 104);
          LODWORD(v22[0]) = v14;
          v15 = *(_BYTE **)(a3 + 8);
          if ( v15 == *(_BYTE **)(a3 + 16) )
          {
            sub_B8BBF0(a3, v15, v22);
            v9 = a1 + 8;
          }
          else
          {
            if ( v15 )
            {
              *(_DWORD *)v15 = v14;
              v15 = *(_BYTE **)(a3 + 8);
            }
            *(_QWORD *)(a3 + 8) = v15 + 4;
          }
        }
        goto LABEL_5;
      case 377:
        result = sub_388BFE0(a1, a2);
        if ( (_BYTE)result )
          return result;
        v8 = *(_DWORD *)(a1 + 64);
        continue;
      default:
        result = v20;
        if ( a4 )
        {
          v16 = *(_QWORD *)(a1 + 56);
          v24 = 1;
          v23 = 3;
          v22[0] = "unterminated attribute group";
          return sub_38814C0(a1 + 8, v16, (__int64)v22);
        }
        return result;
    }
  }
}
