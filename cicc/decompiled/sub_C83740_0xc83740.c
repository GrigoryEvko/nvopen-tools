// Function: sub_C83740
// Address: 0xc83740
//
__int64 __fastcall sub_C83740(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  _OWORD v10[2]; // [rsp+0h] [rbp-30h] BYREF

  memset(v10, 0, sizeof(v10));
  LOWORD(v10[0]) = 1;
  if ( fcntl(a1, 7, v10, a4, a5, a6) == -1 )
  {
    sub_2241E50(a1, 7, v6, v7, v8);
    return (unsigned int)*__errno_location();
  }
  else
  {
    sub_2241E40(a1, 7, v6, v7, v8);
    return 0;
  }
}
