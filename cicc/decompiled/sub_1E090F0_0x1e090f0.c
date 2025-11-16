// Function: sub_1E090F0
// Address: 0x1e090f0
//
__int64 __fastcall sub_1E090F0(__int64 a1, __int64 a2, unsigned int a3, char a4, __int64 a5, char a6)
{
  unsigned int v6; // r13d
  unsigned int v7; // r14d
  __m128i v9; // [rsp+0h] [rbp-50h] BYREF
  unsigned int v10; // [rsp+10h] [rbp-40h]
  char v11; // [rsp+14h] [rbp-3Ch]
  char v12; // [rsp+15h] [rbp-3Bh]
  char v13; // [rsp+16h] [rbp-3Ah]
  char v14; // [rsp+17h] [rbp-39h]
  __int64 v15; // [rsp+18h] [rbp-38h]
  char v16; // [rsp+20h] [rbp-30h]
  char v17; // [rsp+21h] [rbp-2Fh]
  __int16 v18; // [rsp+22h] [rbp-2Eh]
  char v19; // [rsp+24h] [rbp-2Ch]

  v6 = *(_DWORD *)a1;
  if ( *(_BYTE *)(a1 + 4) || v6 >= a3 )
    v6 = a3;
  v9.m128i_i64[1] = a2;
  v12 = a4;
  v10 = v6;
  v14 = a6;
  v15 = a5;
  v17 = a4 ^ 1;
  v18 = 0;
  v9.m128i_i64[0] = 0;
  v11 = 0;
  v13 = 0;
  v16 = 0;
  v19 = 0;
  sub_1E090A0(a1 + 8, &v9);
  v7 = -858993459 * ((__int64)(*(_QWORD *)(a1 + 16) - *(_QWORD *)(a1 + 8)) >> 3) + ~*(_DWORD *)(a1 + 32);
  sub_1E08740(a1, v6);
  return v7;
}
