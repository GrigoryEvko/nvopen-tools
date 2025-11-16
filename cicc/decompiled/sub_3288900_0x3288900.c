// Function: sub_3288900
// Address: 0x3288900
//
__int64 __fastcall sub_3288900(__int64 a1, unsigned int a2, __int64 a3, int a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  __int128 v7; // [rsp-10h] [rbp-40h]
  __int64 v8; // [rsp+0h] [rbp-30h]
  __int64 v9; // [rsp+10h] [rbp-20h] BYREF
  int v10; // [rsp+18h] [rbp-18h]

  if ( *(_DWORD *)(a5 + 24) == 51 )
  {
    v9 = 0;
    v10 = 0;
    result = sub_33F17F0(a1, 51, &v9, a2, a3);
    if ( v9 )
    {
      v8 = result;
      sub_B91220((__int64)&v9, v9);
      return v8;
    }
  }
  else
  {
    *((_QWORD *)&v7 + 1) = a6;
    *(_QWORD *)&v7 = a5;
    return sub_33FAF80(a1, 168, a4, a2, a3, a6, v7);
  }
  return result;
}
