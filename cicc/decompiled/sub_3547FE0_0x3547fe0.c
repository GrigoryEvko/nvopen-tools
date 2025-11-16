// Function: sub_3547FE0
// Address: 0x3547fe0
//
void __fastcall sub_3547FE0(__int64 *a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v9; // rsi
  unsigned __int64 v10; // rsi
  __int64 v11; // rbx
  __int64 v12; // r13
  __int64 v13; // rsi

  *a1 = a3;
  a1[23] = (__int64)(a1 + 25);
  a1[5] = (__int64)(a1 + 7);
  a1[41] = (__int64)(a1 + 43);
  a1[2] = 0;
  a1[3] = 0;
  a1[4] = 0;
  a1[6] = 0x400000000LL;
  a1[24] = 0x400000000LL;
  a1[42] = 0x400000000LL;
  a1[59] = (__int64)(a1 + 61);
  a1[60] = 0x400000000LL;
  v9 = a2[1] - *a2;
  a1[1] = a4;
  v10 = v9 >> 8;
  if ( v10 )
    sub_3547CE0(a1 + 2, v10, (__int64)(a1 + 61), a4, a5, a6);
  sub_3545870(a1, a3);
  sub_3545870(a1, a4);
  v11 = *a2;
  v12 = a2[1];
  while ( v12 != v11 )
  {
    v13 = v11;
    v11 += 256;
    sub_3545870(a1, v13);
  }
}
