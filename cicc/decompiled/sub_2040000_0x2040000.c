// Function: sub_2040000
// Address: 0x2040000
//
__int64 __fastcall sub_2040000(__int64 a1, unsigned __int64 a2, unsigned int a3, __m128i a4, double a5, __m128i a6)
{
  unsigned int *v7; // rax
  __int64 v8; // rax
  unsigned int v9; // eax
  __int64 v10; // r8
  __int64 v11; // r9
  unsigned int v12; // r13d
  __int16 v13; // ax
  unsigned int v14; // edx
  __int64 *v15; // rcx
  const __m128i *v16; // r9
  unsigned __int64 v17; // r8
  unsigned int v19; // edx
  unsigned int v20; // edx
  unsigned int v21; // edx
  unsigned int v22; // edx
  unsigned int v23; // edx
  unsigned int v24; // edx
  unsigned int v25; // edx
  unsigned int v26; // edx
  unsigned int v27; // edx
  __int64 v28; // [rsp-8h] [rbp-D0h]
  __int64 v29; // [rsp+0h] [rbp-C8h]

  v7 = (unsigned int *)(*(_QWORD *)(a2 + 32) + 40LL * a3);
  v8 = *(_QWORD *)(*(_QWORD *)v7 + 40LL) + 16LL * v7[2];
  v9 = sub_2016240((_QWORD *)a1, a2, *(_BYTE *)v8, *(_QWORD *)(v8 + 8), 0, 0, 0);
  if ( (_BYTE)v9 )
    return 0;
  v12 = v9;
  v13 = *(_WORD *)(a2 + 24);
  if ( v13 > 186 )
  {
    v15 = v13 == 236
        ? (__int64 *)sub_203DB10((__int64 *)a1, a2, a3, *(double *)a4.m128i_i64, a5, a6)
        : sub_203DF40((__int64 *)a1, a2, a4, a5, a6);
    v17 = v14;
  }
  else
  {
    switch ( v13 )
    {
      case 101:
        v15 = sub_1D40890(*(__int64 **)(a1 + 8), a2, 0, v29, v10, v11, a4, a5, a6);
        v17 = v23;
        break;
      case 102:
      case 103:
      case 104:
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
      case 134:
      case 135:
      case 136:
      case 138:
      case 139:
      case 140:
      case 141:
      case 148:
      case 149:
      case 150:
      case 151:
      case 154:
      case 155:
      case 156:
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
      case 186:
        v15 = sub_203FDC0((__int64 *)a1, a2, *(double *)a4.m128i_i64, a5, a6);
        v17 = v22;
        break;
      case 106:
        v15 = sub_203DA30(a1, a2, *(double *)a4.m128i_i64, a5, a6);
        v17 = v24;
        break;
      case 107:
        v15 = sub_203D440((__int64 **)a1, a2, a4, a5, a6, v28, v29, v10, v11);
        v17 = v25;
        break;
      case 109:
        v15 = sub_203D950(a1, a2, *(double *)a4.m128i_i64, a5, a6);
        v17 = v26;
        break;
      case 137:
        v15 = (__int64 *)sub_203E430((__int64 *)a1, a2, (__m128)a4, a5, a6);
        v17 = v27;
        break;
      case 142:
      case 143:
      case 144:
        v15 = (__int64 *)sub_203CAD0((__int64 **)a1, a2, *(double *)a4.m128i_i64, a5, a6);
        v17 = v20;
        break;
      case 145:
      case 146:
      case 147:
      case 152:
      case 153:
      case 157:
        v15 = sub_203C550((__int64 **)a1, a2, *(double *)a4.m128i_i64, a5, a6);
        v17 = v19;
        break;
      case 158:
        v15 = (__int64 *)sub_203D1B0((__int64 **)a1, a2, a4, a5, a6);
        v17 = v21;
        break;
    }
  }
  if ( !v15 )
  {
    return 0;
  }
  else if ( (__int64 *)a2 == v15 )
  {
    return 1;
  }
  else
  {
    sub_2013400(a1, a2, 0, (__int64)v15, (__m128i *)v17, v16);
  }
  return v12;
}
