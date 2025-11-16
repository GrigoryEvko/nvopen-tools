// Function: sub_211F5C0
// Address: 0x211f5c0
//
__int64 __fastcall sub_211F5C0(__int64 *a1, unsigned __int64 a2, unsigned int a3, __m128i a4, __m128i a5, __m128i a6)
{
  unsigned int *v6; // rax
  __int64 v7; // rax
  unsigned int v8; // eax
  unsigned int v9; // r15d
  __int16 v10; // ax
  unsigned int v11; // edx
  __int64 v12; // rcx
  const __m128i *v13; // r9
  unsigned __int64 v14; // r8
  unsigned int v16; // edx
  unsigned int v17; // edx
  unsigned int v18; // edx
  unsigned int v19; // edx
  unsigned int v20; // edx
  unsigned int v21; // edx
  unsigned int v22; // edx
  unsigned int v23; // edx
  unsigned int v24; // edx
  __int64 v25; // [rsp-8h] [rbp-E0h]

  v6 = (unsigned int *)(*(_QWORD *)(a2 + 32) + 40LL * a3);
  v7 = *(_QWORD *)(*(_QWORD *)v6 + 40LL) + 16LL * v6[2];
  v8 = sub_2016240(a1, a2, *(_BYTE *)v7, *(_QWORD *)(v7 + 8), 0, 0, 0);
  if ( (_BYTE)v8 )
    return 0;
  v9 = v8;
  v10 = *(_WORD *)(a2 + 24);
  if ( v10 <= 135 )
  {
    if ( v10 == 101 )
    {
      v12 = (__int64)sub_211EDB0((__int64)a1, a2, *(double *)a4.m128i_i64, *(double *)a5.m128i_i64, a6);
      v14 = v16;
    }
    else
    {
      v12 = v10 == 104 ? sub_21459C0(a1, a2, v25) : sub_2145E30(a1, a2, v25);
      v14 = v11;
    }
  }
  else
  {
    switch ( v10 )
    {
      case 136:
        v12 = (__int64)sub_211F150((__int64)a1, a2, *(double *)a4.m128i_i64, *(double *)a5.m128i_i64, a6);
        v14 = v21;
        break;
      case 137:
        v12 = (__int64)sub_211F2E0((__int64)a1, a2, *(double *)a4.m128i_i64, *(double *)a5.m128i_i64, a6);
        v14 = v22;
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
      case 192:
        v12 = (__int64)sub_211EC20((__int64)a1, a2, *(double *)a4.m128i_i64, *(double *)a5.m128i_i64, a6);
        v14 = v18;
        break;
      case 152:
        v12 = sub_211EF90((__int64)a1, a2, a4, a5, a6);
        v14 = v23;
        break;
      case 153:
        v12 = sub_211F070((__int64)a1, a2, a4, a5, a6);
        v14 = v24;
        break;
      case 154:
        v12 = (__int64)sub_211EEA0((__int64)a1, a2, *(double *)a4.m128i_i64, *(double *)a5.m128i_i64, a6);
        v14 = v19;
        break;
      case 158:
        v12 = sub_2145520(a1, a2, v25);
        v14 = v20;
        break;
      case 186:
        v12 = sub_211F3C0(a1, a2);
        v14 = v17;
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
    sub_2013400((__int64)a1, a2, 0, v12, (__m128i *)v14, v13);
  }
  return v9;
}
