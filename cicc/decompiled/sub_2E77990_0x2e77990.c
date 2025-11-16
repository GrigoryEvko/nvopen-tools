// Function: sub_2E77990
// Address: 0x2e77990
//
__int64 __fastcall sub_2E77990(__int64 a1, __int64 a2, __int64 a3, char a4, char a5)
{
  __int64 v7; // rax
  unsigned __int8 v8; // cl
  unsigned __int8 v9; // si
  unsigned __int64 v10; // rax
  __m128i *v11; // rsi
  int v12; // eax
  int v13; // edx
  __m128i v15; // [rsp+0h] [rbp-40h] BYREF
  unsigned __int8 v16; // [rsp+10h] [rbp-30h]
  char v17; // [rsp+11h] [rbp-2Fh]
  __int16 v18; // [rsp+12h] [rbp-2Eh]
  char v19; // [rsp+14h] [rbp-2Ch]
  __int64 v20; // [rsp+18h] [rbp-28h]
  char v21; // [rsp+20h] [rbp-20h]
  char v22; // [rsp+21h] [rbp-1Fh]
  __int16 v23; // [rsp+22h] [rbp-1Eh]
  char v24; // [rsp+24h] [rbp-1Ch]

  v7 = 1;
  v8 = *(_BYTE *)a1;
  if ( !*(_BYTE *)(a1 + 2) )
    v7 = 1LL << v8;
  v9 = -1;
  v10 = (a3 | v7) & -(a3 | v7);
  if ( v10 )
  {
    _BitScanReverse64(&v10, v10);
    v9 = 63 - (v10 ^ 0x3F);
  }
  if ( !*(_BYTE *)(a1 + 1) && v9 > v8 )
    v9 = *(_BYTE *)a1;
  v16 = v9;
  v11 = *(__m128i **)(a1 + 8);
  v15.m128i_i64[0] = a3;
  v23 = 0;
  v15.m128i_i64[1] = a2;
  v17 = a4;
  v18 = 0;
  v19 = 0;
  v20 = 0;
  v21 = 0;
  v22 = a5;
  v24 = 0;
  sub_2E77880((unsigned __int64 *)(a1 + 8), v11, &v15);
  v12 = *(_DWORD *)(a1 + 32);
  v13 = v12 + 1;
  *(_DWORD *)(a1 + 32) = v13;
  return (unsigned int)~v12;
}
