// Function: sub_2E77BD0
// Address: 0x2e77bd0
//
__int64 __fastcall sub_2E77BD0(__int64 a1, __int64 a2, unsigned __int8 a3, unsigned __int8 a4, __int64 a5, char a6)
{
  unsigned __int8 v6; // r13
  unsigned int v8; // r14d
  __m128i v10; // [rsp+0h] [rbp-50h] BYREF
  unsigned __int8 v11; // [rsp+10h] [rbp-40h]
  char v12; // [rsp+11h] [rbp-3Fh]
  unsigned __int8 v13; // [rsp+12h] [rbp-3Eh]
  char v14; // [rsp+13h] [rbp-3Dh]
  char v15; // [rsp+14h] [rbp-3Ch]
  __int64 v16; // [rsp+18h] [rbp-38h]
  char v17; // [rsp+20h] [rbp-30h]
  __int16 v18; // [rsp+21h] [rbp-2Fh]
  char v19; // [rsp+23h] [rbp-2Dh]
  char v20; // [rsp+24h] [rbp-2Ch]

  v6 = a3;
  if ( a3 > (unsigned int)*(_WORD *)a1 )
    v6 = *(_BYTE *)a1;
  v10.m128i_i64[1] = a2;
  v13 = a4;
  v15 = a6;
  v19 = 0;
  v10.m128i_i64[0] = 0;
  v11 = v6;
  v12 = 0;
  v14 = 0;
  v16 = a5;
  v17 = 0;
  v18 = a4 ^ 1;
  v20 = 0;
  sub_2E77AF0((unsigned __int64 *)(a1 + 8), &v10);
  v8 = -858993459 * ((__int64)(*(_QWORD *)(a1 + 16) - *(_QWORD *)(a1 + 8)) >> 3) + ~*(_DWORD *)(a1 + 32);
  if ( (a6 & 0xFD) == 0 )
    sub_2E76F70(a1, v6);
  return v8;
}
