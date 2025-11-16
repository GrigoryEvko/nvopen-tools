// Function: sub_7BE1A0
// Address: 0x7be1a0
//
__int64 __fastcall sub_7BE1A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _BYTE v7[352]; // [rsp+0h] [rbp-170h] BYREF
  int v8; // [rsp+160h] [rbp-10h]
  __int16 v9; // [rsp+164h] [rbp-Ch]

  memset(v7, 0, sizeof(v7));
  *(_WORD *)&v7[9] = 257;
  v7[74] = 1;
  v7[28] = 1;
  v8 = 0;
  v9 = 0;
  return sub_7BDFF0((unsigned __int64)v7, 0, 257, 0, (__int64)v7, a6);
}
