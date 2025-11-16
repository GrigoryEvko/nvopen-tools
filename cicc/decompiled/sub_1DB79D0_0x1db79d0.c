// Function: sub_1DB79D0
// Address: 0x1db79d0
//
__int64 __fastcall sub_1DB79D0(__int64 *a1, __int64 a2, __int64 *a3)
{
  bool v3; // zf
  __int64 *v5; // [rsp+8h] [rbp-8h] BYREF

  v3 = a1[12] == 0;
  v5 = a1;
  if ( v3 )
    return sub_1DB6E70(&v5, a2, a3, 0);
  else
    return sub_1DB75E0((__int64 *)&v5, a2, a3, 0);
}
