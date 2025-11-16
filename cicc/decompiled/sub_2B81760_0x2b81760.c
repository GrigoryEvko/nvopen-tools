// Function: sub_2B81760
// Address: 0x2b81760
//
__int64 __fastcall sub_2B81760(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v7; // r9
  char **v8; // r15
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // [rsp-10h] [rbp-50h]
  __int64 v14; // [rsp+8h] [rbp-38h]

  v7 = ((((a2 - a1) >> 6) + 1) / 2) << 6;
  v14 = v7;
  v8 = (char **)(a1 + v7);
  if ( (((a2 - a1) >> 6) + 1) / 2 <= a4 )
  {
    sub_2B0FED0(a1, a1 + v7, a3, a4, a5, v7);
    sub_2B0FED0((__int64)v8, a2, a3, v10, v11, v12);
  }
  else
  {
    sub_2B81760(a1, a1 + v7, a3);
    sub_2B81760(v8, a2, a3);
  }
  sub_2B812F0(a1, v8, a2, v14 >> 6, (a2 - (__int64)v8) >> 6, a3, a4);
  return v13;
}
