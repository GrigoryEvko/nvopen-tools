// Function: sub_8958E0
// Address: 0x8958e0
//
__int64 __fastcall sub_8958E0(__int64 *a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 result; // rax
  __int64 v5; // rdi
  char v6; // al
  __int64 v7; // rax
  __int64 v8; // rdx
  __m128i *v9; // r13
  __int64 *v10; // r15
  bool v11; // r14
  __int64 v12; // rsi
  char v13; // dl
  __int8 v14; // r14
  __int64 v15; // [rsp+0h] [rbp-60h]
  __int64 v16; // [rsp+8h] [rbp-58h]
  __int64 v17; // [rsp+10h] [rbp-50h]
  __int64 v18; // [rsp+18h] [rbp-48h]
  __int8 v19; // [rsp+27h] [rbp-39h]
  __int64 v20; // [rsp+28h] [rbp-38h]

  v2 = *a1;
  result = *(_QWORD *)(*a1 + 288);
  if ( *(_BYTE *)(result + 140) != 7 )
    return result;
  result = *(_QWORD *)(result + 168);
  if ( !result )
    return result;
  v5 = *(_QWORD *)(result + 56);
  if ( !v5 || (*(_BYTE *)v5 & 0x20) == 0 )
    return result;
  v6 = *(_BYTE *)(a2 + 80);
  if ( (unsigned __int8)(v6 - 19) > 3u )
  {
LABEL_12:
    switch ( v6 )
    {
      case 4:
      case 5:
        v9 = *(__m128i **)(*(_QWORD *)(a2 + 96) + 80LL);
        goto LABEL_10;
      case 6:
        v9 = *(__m128i **)(*(_QWORD *)(a2 + 96) + 32LL);
        goto LABEL_10;
      case 9:
      case 10:
        v9 = *(__m128i **)(*(_QWORD *)(a2 + 96) + 56LL);
        goto LABEL_10;
      case 19:
      case 20:
      case 21:
      case 22:
        goto LABEL_9;
      default:
        BUG();
    }
  }
  v7 = *(_QWORD *)(a2 + 88);
  v8 = *(_QWORD *)(v7 + 88);
  if ( v8 && (*(_BYTE *)(v7 + 160) & 1) == 0 )
  {
    v6 = *(_BYTE *)(v8 + 80);
    a2 = v8;
    goto LABEL_12;
  }
LABEL_9:
  v9 = *(__m128i **)(a2 + 88);
LABEL_10:
  v10 = (__int64 *)v9[11].m128i_i64[0];
  v16 = v9[21].m128i_i64[0];
  v17 = v9[21].m128i_i64[1];
  v15 = v10[19];
  v18 = v9[22].m128i_i64[0];
  v19 = v9[22].m128i_i8[8];
  v20 = v9[23].m128i_i64[0];
  v11 = (v9[26].m128i_i8[8] & 4) != 0;
  sub_879080(v9 + 21, *(const __m128i **)(v5 + 8), a1[24]);
  v9[26].m128i_i8[8] &= ~4u;
  v12 = *v10;
  v10[19] = *(_QWORD *)(v2 + 288);
  sub_894C10((__int64)a1, v12);
  v13 = 4 * v11;
  v10[19] = v15;
  v14 = v9[26].m128i_i8[8];
  v9[21].m128i_i64[0] = v16;
  v9[21].m128i_i64[1] = v17;
  v9[22].m128i_i64[0] = v18;
  v9[22].m128i_i8[8] = v19;
  v9[26].m128i_i8[8] = v13 | v14 & 0xFB;
  v9[23].m128i_i64[0] = v20;
  return v20;
}
