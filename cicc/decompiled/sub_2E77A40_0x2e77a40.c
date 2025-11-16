// Function: sub_2E77A40
// Address: 0x2e77a40
//
__int64 __fastcall sub_2E77A40(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  __int64 v6; // rax
  unsigned __int8 v7; // cl
  unsigned __int8 v8; // si
  unsigned __int64 v9; // rax
  __m128i *v10; // rsi
  int v11; // eax
  int v12; // edx
  __m128i v14; // [rsp+0h] [rbp-40h] BYREF
  unsigned __int8 v15; // [rsp+10h] [rbp-30h]
  char v16; // [rsp+11h] [rbp-2Fh]
  __int16 v17; // [rsp+12h] [rbp-2Eh]
  char v18; // [rsp+14h] [rbp-2Ch]
  __int64 v19; // [rsp+18h] [rbp-28h]
  int v20; // [rsp+20h] [rbp-20h]
  char v21; // [rsp+24h] [rbp-1Ch]

  v6 = 1;
  v7 = *(_BYTE *)a1;
  if ( !*(_BYTE *)(a1 + 2) )
    v6 = 1LL << v7;
  v8 = -1;
  v9 = (a3 | v6) & -(a3 | v6);
  if ( v9 )
  {
    _BitScanReverse64(&v9, v9);
    v8 = 63 - (v9 ^ 0x3F);
  }
  if ( !*(_BYTE *)(a1 + 1) && v8 > v7 )
    v8 = *(_BYTE *)a1;
  v15 = v8;
  v10 = *(__m128i **)(a1 + 8);
  v14.m128i_i64[0] = a3;
  v14.m128i_i64[1] = a2;
  v16 = a4;
  v17 = 1;
  v18 = 0;
  v19 = 0;
  v20 = 0;
  v21 = 0;
  sub_2E77880((unsigned __int64 *)(a1 + 8), v10, &v14);
  v11 = *(_DWORD *)(a1 + 32);
  v12 = v11 + 1;
  *(_DWORD *)(a1 + 32) = v12;
  return (unsigned int)~v11;
}
