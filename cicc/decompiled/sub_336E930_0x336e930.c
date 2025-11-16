// Function: sub_336E930
// Address: 0x336e930
//
__int64 __fastcall sub_336E930(__int64 a1, __int64 a2, __int64 a3, int a4, __int64 a5, int a6)
{
  __int64 v10; // r14
  __int64 v11; // [rsp+0h] [rbp-50h] BYREF
  __int64 v12; // [rsp+8h] [rbp-48h]
  __int64 v13; // [rsp+10h] [rbp-40h] BYREF
  int v14; // [rsp+18h] [rbp-38h]

  v11 = a2;
  v12 = a3;
  if ( (_WORD)a2 )
  {
    if ( (unsigned __int16)(a2 - 176) > 0x34u )
      return sub_32886A0(a1, (unsigned int)v11, v12, a4, a5, a6);
  }
  else if ( !sub_3007100((__int64)&v11) )
  {
    return sub_32886A0(a1, (unsigned int)v11, v12, a4, a5, a6);
  }
  if ( *(_DWORD *)(a5 + 24) != 51 )
    return sub_33FAF80(a1, 168, a4, v11, v12, a6);
  v13 = 0;
  v14 = 0;
  v10 = sub_33F17F0(a1, 51, &v13, v11, v12);
  if ( v13 )
    sub_B91220((__int64)&v13, v13);
  return v10;
}
