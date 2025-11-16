// Function: sub_1E09450
// Address: 0x1e09450
//
__int64 __fastcall sub_1E09450(unsigned int *a1, __int64 a2, __int64 a3, char a4)
{
  __int64 v4; // rax
  unsigned int v8; // ecx
  __int64 v9; // rax
  __m128i *v10; // rsi
  unsigned int v11; // eax
  unsigned int v12; // edx
  __m128i v14; // [rsp+0h] [rbp-40h] BYREF
  unsigned int v15; // [rsp+10h] [rbp-30h]
  char v16; // [rsp+14h] [rbp-2Ch]
  __int64 v17; // [rsp+15h] [rbp-2Bh]
  __int64 v18; // [rsp+1Dh] [rbp-23h]

  v4 = 1;
  v8 = *a1;
  if ( !*((_BYTE *)a1 + 5) )
    v4 = *a1;
  v9 = -(a3 | v4) & (a3 | v4);
  if ( *((_BYTE *)a1 + 4) || v8 >= (unsigned int)v9 )
    v8 = v9;
  v10 = (__m128i *)*((_QWORD *)a1 + 1);
  v14.m128i_i64[0] = a3;
  v16 = a4;
  v14.m128i_i64[1] = a2;
  v15 = v8;
  v17 = 1;
  v18 = 0;
  sub_1E092B0((__int64)(a1 + 2), v10, &v14);
  v11 = a1[8];
  v12 = v11 + 1;
  a1[8] = v12;
  return ~v11;
}
