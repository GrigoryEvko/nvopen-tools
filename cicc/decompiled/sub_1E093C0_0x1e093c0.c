// Function: sub_1E093C0
// Address: 0x1e093c0
//
__int64 __fastcall sub_1E093C0(unsigned int *a1, __int64 a2, __int64 a3, char a4, char a5)
{
  __int64 v6; // rax
  unsigned int v9; // ecx
  __int64 v10; // rax
  __m128i *v11; // rsi
  unsigned int v12; // eax
  unsigned int v13; // edx
  __m128i v15; // [rsp+0h] [rbp-40h] BYREF
  unsigned int v16; // [rsp+10h] [rbp-30h]
  char v17; // [rsp+14h] [rbp-2Ch]
  __int64 v18; // [rsp+15h] [rbp-2Bh]
  int v19; // [rsp+1Dh] [rbp-23h]
  char v20; // [rsp+21h] [rbp-1Fh]
  __int16 v21; // [rsp+22h] [rbp-1Eh]
  char v22; // [rsp+24h] [rbp-1Ch]

  v6 = 1;
  v9 = *a1;
  if ( !*((_BYTE *)a1 + 5) )
    v6 = *a1;
  v10 = -(a3 | v6) & (a3 | v6);
  if ( *((_BYTE *)a1 + 4) || v9 >= (unsigned int)v10 )
    v9 = v10;
  v11 = (__m128i *)*((_QWORD *)a1 + 1);
  v15.m128i_i64[0] = a3;
  v15.m128i_i64[1] = a2;
  v16 = v9;
  v17 = a4;
  v18 = 0;
  v19 = 0;
  v20 = a5;
  v21 = 0;
  v22 = 0;
  sub_1E092B0((__int64)(a1 + 2), v11, &v15);
  v12 = a1[8];
  v13 = v12 + 1;
  a1[8] = v13;
  return ~v12;
}
