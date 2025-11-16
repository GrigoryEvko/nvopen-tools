// Function: sub_2126F30
// Address: 0x2126f30
//
__int64 __fastcall sub_2126F30(__int64 *a1, unsigned __int64 a2, __m128 a3, double a4, __m128i a5)
{
  __int16 v5; // ax
  __int64 v6; // rcx
  const __m128i *v7; // r9
  unsigned int v8; // edx
  unsigned __int64 v9; // r8
  unsigned int v11; // edx
  unsigned int v12; // edx
  unsigned int v13; // edx
  unsigned int v14; // edx
  unsigned int v15; // edx
  unsigned int v16; // edx

  v5 = *(_WORD *)(a2 + 24);
  if ( v5 == 101 )
  {
    v6 = (__int64)sub_2126820((__int64)a1, a2, *(double *)a3.m128_u64, a4, a5);
    v9 = v16;
  }
  else
  {
    switch ( v5 )
    {
      case 136:
        v6 = (__int64)sub_2126AB0((__int64)a1, a2);
        v9 = v14;
        break;
      case 137:
        v6 = (__int64)sub_2126BC0(a1, a2, a3, a4, a5);
        v9 = v15;
        break;
      case 138:
      case 139:
      case 140:
      case 141:
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
        v6 = sub_2126D00((__int64)a1, a2, *(double *)a3.m128_u64, a4, *(double *)a5.m128i_i64);
        v9 = v13;
        break;
      case 152:
      case 153:
        v6 = sub_2126900((__int64)a1, a2, *(double *)a3.m128_u64, a4, *(double *)a5.m128i_i64);
        v9 = v8;
        break;
      case 157:
        v6 = sub_21269C0((__int64)a1, a2, *(double *)a3.m128_u64, a4, *(double *)a5.m128i_i64);
        v9 = v11;
        break;
      case 158:
        v6 = sub_2126630((__int64)a1, a2, *(double *)a3.m128_u64, a4, *(double *)a5.m128i_i64);
        v9 = v12;
        break;
    }
  }
  if ( v6 )
    sub_2013400((__int64)a1, a2, 0, v6, (__m128i *)v9, v7);
  return 0;
}
