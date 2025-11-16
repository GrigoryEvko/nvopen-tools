// Function: sub_1DB7A00
// Address: 0x1db7a00
//
__int64 __fastcall sub_1DB7A00(__int64 *a1, __int64 a2)
{
  bool v3; // zf
  __int64 v4; // rsi
  __int64 *v6; // [rsp+8h] [rbp-8h] BYREF

  v3 = a1[12] == 0;
  v6 = a1;
  v4 = *(_QWORD *)(a2 + 8);
  if ( v3 )
    return sub_1DB6E70(&v6, v4, 0, a2);
  else
    return sub_1DB75E0((__int64 *)&v6, v4, 0, a2);
}
