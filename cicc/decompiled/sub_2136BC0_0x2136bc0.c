// Function: sub_2136BC0
// Address: 0x2136bc0
//
__int64 __fastcall sub_2136BC0(__int64 a1, unsigned __int64 a2, unsigned int a3, double a4, __m128i a5, __m128i a6)
{
  unsigned int *v6; // rax
  __int64 v7; // rax
  unsigned int v8; // eax
  unsigned int v9; // r15d
  __int16 v10; // ax
  unsigned __int64 v11; // rbx
  __int64 v12; // rcx
  const __m128i *v13; // r9
  unsigned int v14; // edx
  unsigned __int64 *v16; // rax
  __int64 v17; // rdx
  unsigned int v18; // edx
  unsigned int v19; // edx
  unsigned int v20; // edx
  unsigned int v21; // edx
  unsigned int v22; // edx
  unsigned int v23; // edx
  unsigned int v24; // edx
  unsigned int v25; // edx
  unsigned int v26; // edx
  unsigned int v27; // edx
  unsigned int v28; // edx
  unsigned int v29; // edx
  unsigned int v30; // edx
  unsigned int v31; // edx
  __int64 v32; // [rsp-8h] [rbp-150h]
  __int64 v33; // [rsp+F8h] [rbp-50h] BYREF
  __int64 v34; // [rsp+100h] [rbp-48h]
  __int64 v35; // [rsp+108h] [rbp-40h] BYREF
  int v36; // [rsp+110h] [rbp-38h]

  v6 = (unsigned int *)(*(_QWORD *)(a2 + 32) + 40LL * a3);
  v7 = *(_QWORD *)(*(_QWORD *)v6 + 40LL) + 16LL * v6[2];
  v8 = sub_2016240((_QWORD *)a1, a2, *(_BYTE *)v7, *(_QWORD *)(v7 + 8), 0, 0, 0);
  if ( (_BYTE)v8 )
    return 0;
  v9 = v8;
  v10 = *(_WORD *)(a2 + 24);
  if ( v10 <= 103 )
  {
    v11 = 0;
    if ( v10 <= 21 )
    {
      v16 = *(unsigned __int64 **)(a2 + 32);
      LODWORD(v34) = 0;
      v36 = 0;
      v17 = v16[1];
      v33 = 0;
      v35 = 0;
      sub_20174B0(a1, *v16, v17, &v33, &v35);
      v12 = (__int64)sub_1D2DE40(*(_QWORD **)(a1 + 8), (__int64 *)a2, v33, v34);
    }
    else
    {
      v12 = sub_2145E30(a1, a2, v32);
      v11 = v14;
    }
  }
  else
  {
    switch ( v10 )
    {
      case 104:
        v12 = sub_21459C0(a1, a2, v32);
        v11 = v28;
        break;
      case 105:
        v12 = sub_2145F30(a1, a2, v32);
        v11 = v24;
        break;
      case 106:
      case 107:
      case 108:
      case 109:
      case 110:
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
      case 127:
      case 128:
      case 129:
      case 130:
      case 131:
      case 132:
      case 133:
      case 134:
      case 135:
      case 139:
      case 140:
      case 141:
      case 142:
      case 143:
      case 144:
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
      case 185:
      case 187:
      case 188:
      case 189:
      case 190:
      case 191:
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
      case 204:
      case 205:
      case 206:
      case 207:
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
        v12 = (__int64)sub_2136B20(a1, a2);
        v11 = v27;
        break;
      case 111:
        v12 = sub_21463F0(a1, a2, v32);
        v11 = v30;
        break;
      case 122:
      case 123:
      case 124:
      case 125:
      case 126:
        v12 = (__int64)sub_21351B0(a1, a2);
        v11 = v18;
        break;
      case 136:
        v12 = (__int64)sub_2134D40(a1, a2, a4, *(double *)a5.m128i_i64, a6);
        v11 = v26;
        break;
      case 137:
        v12 = (__int64)sub_2134ED0(a1, a2, a4, *(double *)a5.m128i_i64, a6);
        v11 = v31;
        break;
      case 138:
        v12 = (__int64)sub_2134FB0(a1, a2);
        v11 = v21;
        break;
      case 145:
        v12 = sub_21361E0(a1, a2, a4, *(double *)a5.m128i_i64, *(double *)a6.m128i_i64);
        v11 = v20;
        break;
      case 146:
        v12 = sub_2135230(a1, a2, a4, a5, a6);
        v11 = v19;
        break;
      case 147:
        v12 = (__int64)sub_21362C0((__int64 *)a1, (_QWORD *)a2, a4, a5, a6);
        v11 = v25;
        break;
      case 158:
        v12 = sub_2145520(a1, a2, v32);
        v11 = v23;
        break;
      case 186:
        v12 = sub_2135330(a1, a2);
        v11 = v22;
        break;
      case 192:
        v12 = (__int64)sub_2134BB0(a1, a2, a4, *(double *)a5.m128i_i64, a6);
        v11 = v29;
        break;
    }
  }
  if ( !v12 )
  {
    return 0;
  }
  else if ( a2 == v12 )
  {
    return 1;
  }
  else
  {
    sub_2013400(a1, a2, 0, v12, (__m128i *)v11, v13);
  }
  return v9;
}
