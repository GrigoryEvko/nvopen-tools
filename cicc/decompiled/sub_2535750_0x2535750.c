// Function: sub_2535750
// Address: 0x2535750
//
__int64 __fastcall sub_2535750(_QWORD **a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rsi
  __int64 v5; // rax
  __int64 v6; // rcx
  unsigned __int64 v7; // rsi
  __m128i v9; // [rsp+0h] [rbp-50h] BYREF
  __int64 v10; // [rsp+10h] [rbp-40h]
  __int64 v11; // [rsp+18h] [rbp-38h]
  __int64 v12; // [rsp+20h] [rbp-30h]
  __int64 v13; // [rsp+28h] [rbp-28h]
  __int64 v14; // [rsp+30h] [rbp-20h]
  __int64 v15; // [rsp+38h] [rbp-18h]
  __int16 v16; // [rsp+40h] [rbp-10h]

  v4 = (*a1)[26];
  v5 = *a1[2];
  v6 = *a1[1];
  v13 = a3;
  v7 = *(_QWORD *)(v4 + 104);
  v12 = v5;
  v9 = (__m128i)v7;
  v10 = 0;
  v11 = v6;
  v14 = 0;
  v15 = 0;
  v16 = 257;
  return (unsigned int)sub_9B6260(a2, &v9, 0) ^ 1;
}
