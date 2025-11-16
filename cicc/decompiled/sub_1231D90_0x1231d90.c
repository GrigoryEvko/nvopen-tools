// Function: sub_1231D90
// Address: 0x1231d90
//
__int64 __fastcall sub_1231D90(__int64 a1, _QWORD *a2, __int64 *a3)
{
  __int16 v3; // r14
  __int64 v4; // r13
  int v6; // eax
  unsigned int v7; // eax
  int v8; // eax
  const char *v9; // rax
  unsigned __int64 v10; // rsi
  __int64 v12; // rax
  unsigned __int8 v13; // al
  const char *v14; // rax
  int v15; // edi
  char *v16; // rax
  __int64 v17; // rdx
  __int64 v18; // r8
  __int64 v19; // r9
  unsigned __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rdx
  unsigned int v25; // eax
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rdx
  unsigned __int64 v29; // rax
  char v30; // dl
  unsigned __int8 v31; // r13
  _QWORD *v32; // rbx
  _BOOL4 v33; // eax
  unsigned __int64 v34; // [rsp+8h] [rbp-F8h]
  char v35; // [rsp+13h] [rbp-EDh]
  int v36; // [rsp+14h] [rbp-ECh]
  unsigned __int64 v37; // [rsp+18h] [rbp-E8h]
  char v38; // [rsp+28h] [rbp-D8h] BYREF
  char v39; // [rsp+29h] [rbp-D7h] BYREF
  __int16 v40; // [rsp+2Ah] [rbp-D6h] BYREF
  int v41; // [rsp+2Ch] [rbp-D4h] BYREF
  __int64 v42; // [rsp+30h] [rbp-D0h] BYREF
  __int64 v43; // [rsp+38h] [rbp-C8h] BYREF
  __m128i v44; // [rsp+40h] [rbp-C0h] BYREF
  char *v45; // [rsp+50h] [rbp-B0h]
  __int64 v46; // [rsp+58h] [rbp-A8h]
  __int16 v47; // [rsp+60h] [rbp-A0h]
  __m128i v48[2]; // [rsp+70h] [rbp-90h] BYREF
  char v49; // [rsp+90h] [rbp-70h]
  char v50; // [rsp+91h] [rbp-6Fh]
  __m128i v51[2]; // [rsp+A0h] [rbp-60h] BYREF
  char v52; // [rsp+C0h] [rbp-40h]
  char v53; // [rsp+C1h] [rbp-3Fh]

  v3 = 0;
  v4 = a1 + 176;
  v6 = *(_DWORD *)(a1 + 240);
  v38 = 0;
  v41 = 0;
  v39 = 1;
  v40 = 0;
  if ( v6 == 68 )
  {
    v3 = 1;
    v8 = sub_1205200(a1 + 176);
    *(_DWORD *)(a1 + 240) = v8;
    v7 = v8 - 320;
    if ( v7 > 0x1E )
    {
LABEL_5:
      v53 = 1;
      v9 = "expected binary operation in atomicrmw";
LABEL_6:
      v51[0].m128i_i64[0] = (__int64)v9;
      v10 = *(_QWORD *)(a1 + 232);
      v52 = 3;
      sub_11FD800(v4, v10, (__int64)v51, 1);
      return 1;
    }
  }
  else
  {
    v7 = v6 - 320;
  }
  switch ( v7 )
  {
    case 0u:
      v36 = 0;
      v35 = 0;
      break;
    case 1u:
      v36 = 4;
      v35 = 0;
      break;
    case 2u:
      v36 = 7;
      v35 = 0;
      break;
    case 3u:
      v36 = 8;
      v35 = 0;
      break;
    case 4u:
      v36 = 9;
      v35 = 0;
      break;
    case 5u:
      v36 = 10;
      v35 = 0;
      break;
    case 6u:
      v36 = 13;
      v35 = 1;
      break;
    case 7u:
      v36 = 14;
      v35 = 1;
      break;
    case 8u:
      v36 = 15;
      v35 = 0;
      break;
    case 9u:
      v36 = 16;
      v35 = 0;
      break;
    case 0xAu:
      v36 = 17;
      v35 = 0;
      break;
    case 0xBu:
      v36 = 18;
      v35 = 0;
      break;
    case 0xDu:
      v36 = 1;
      v35 = 0;
      break;
    case 0xEu:
      v36 = 11;
      v35 = 1;
      break;
    case 0xFu:
      v36 = 2;
      v35 = 0;
      break;
    case 0x10u:
      v36 = 12;
      v35 = 1;
      break;
    case 0x1Cu:
      v36 = 3;
      v35 = 0;
      break;
    case 0x1Du:
      v36 = 5;
      v35 = 0;
      break;
    case 0x1Eu:
      v36 = 6;
      v35 = 0;
      break;
    default:
      goto LABEL_5;
  }
  *(_DWORD *)(a1 + 240) = sub_1205200(v4);
  v37 = *(_QWORD *)(a1 + 232);
  if ( (unsigned __int8)sub_122FE20((__int64 **)a1, &v42, a3) )
    return 1;
  if ( (unsigned __int8)sub_120AFE0(a1, 4, "expected ',' after atomicrmw address") )
    return 1;
  v34 = *(_QWORD *)(a1 + 232);
  if ( (unsigned __int8)sub_122FE20((__int64 **)a1, &v43, a3)
    || (unsigned __int8)sub_120E4B0(a1, 1u, &v39, &v41)
    || (unsigned __int8)sub_120DF00(a1, &v40, &v38) )
  {
    return 1;
  }
  if ( v41 == 1 )
  {
    v53 = 1;
    v9 = "atomicrmw cannot be unordered";
    goto LABEL_6;
  }
  if ( *(_BYTE *)(*(_QWORD *)(v42 + 8) + 8LL) != 14 )
  {
    v53 = 1;
    v52 = 3;
    v51[0].m128i_i64[0] = (__int64)"atomicrmw operand must be a pointer";
    sub_11FD800(v4, v37, (__int64)v51, 1);
    return 1;
  }
  if ( sub_BCEA30(*(_QWORD *)(v43 + 8)) )
  {
    v53 = 1;
    v52 = 3;
    v51[0].m128i_i64[0] = (__int64)"atomicrmw operand may not be scalable";
    sub_11FD800(v4, v34, (__int64)v51, 1);
    return 1;
  }
  v12 = *(_QWORD *)(v43 + 8);
  if ( !v36 )
  {
    v20 = *(unsigned __int8 *)(v12 + 8);
    if ( (unsigned __int8)v20 > 0xCu || (v21 = 4143, !_bittest64(&v21, v20)) )
    {
      if ( (v20 & 0xFD) != 4 && (_BYTE)v20 != 14 )
      {
        v50 = 1;
        v15 = 0;
        v48[0].m128i_i64[0] = (__int64)" operand must be an integer, floating point, or pointer type";
        v49 = 3;
        goto LABEL_42;
      }
    }
    goto LABEL_48;
  }
  if ( v35 )
  {
    if ( (unsigned int)*(unsigned __int8 *)(v12 + 8) - 17 <= 1 )
      v12 = **(_QWORD **)(v12 + 16);
    v13 = *(_BYTE *)(v12 + 8);
    if ( v13 > 3u && v13 != 5 && (v13 & 0xFD) != 4 )
    {
      v50 = 1;
      v14 = " operand must be a floating point type";
LABEL_41:
      v48[0].m128i_i64[0] = (__int64)v14;
      v15 = v36;
      v49 = 3;
LABEL_42:
      v16 = sub_B4D7D0(v15);
      v47 = 1283;
      v46 = v17;
      v44.m128i_i64[0] = (__int64)"atomicrmw ";
      v45 = v16;
      sub_9C6370(v51, &v44, v48, (__int64)"atomicrmw ", v18, v19);
      sub_11FD800(v4, v34, (__int64)v51, 1);
      return 1;
    }
  }
  else if ( *(_BYTE *)(v12 + 8) != 12 )
  {
    v50 = 1;
    v14 = " operand must be an integer";
    goto LABEL_41;
  }
LABEL_48:
  v22 = sub_B2BEC0(a3[1]);
  v23 = sub_9208B0(v22, *(_QWORD *)(v43 + 8));
  v51[0].m128i_i64[1] = v24;
  v51[0].m128i_i64[0] = (v23 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  v25 = sub_CA1930(v51);
  if ( v25 <= 7 || (v25 & (v25 - 1)) != 0 )
  {
    v53 = 1;
    v52 = 3;
    v51[0].m128i_i64[0] = (__int64)"atomicrmw operand must be power-of-two byte-sized integer";
    sub_11FD800(v4, v34, (__int64)v51, 1);
    return 1;
  }
  else
  {
    v26 = sub_B2BEC0(a3[1]);
    v27 = sub_9208B0(v26, *(_QWORD *)(v43 + 8));
    v51[0].m128i_i64[1] = v28;
    v51[0].m128i_i64[0] = (unsigned __int64)(v27 + 7) >> 3;
    v29 = sub_CA1930(v51);
    v30 = 64;
    if ( v29 )
    {
      _BitScanReverse64(&v29, v29);
      v30 = v29 ^ 0x3F;
    }
    if ( HIBYTE(v40) )
      v31 = v40;
    else
      v31 = 63 - v30;
    v32 = sub_BD2C40(80, unk_3F148C0);
    if ( v32 )
      sub_B4D750((__int64)v32, v36, v42, v43, v31, v41, v39, 0, 0);
    v33 = v38 != 0;
    *((_WORD *)v32 + 1) = *((_WORD *)v32 + 1) & 0xFFFE | v3;
    *a2 = v32;
    return (unsigned int)(2 * v33);
  }
}
