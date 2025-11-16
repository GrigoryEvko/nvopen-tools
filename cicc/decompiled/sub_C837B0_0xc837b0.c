// Function: sub_C837B0
// Address: 0xc837b0
//
__int64 __fastcall sub_C837B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  int v10; // [rsp+0h] [rbp-30h] BYREF
  __int64 v11; // [rsp+8h] [rbp-28h]
  __int64 v12; // [rsp+10h] [rbp-20h]

  v10 = 2;
  v11 = 0;
  v12 = 0;
  if ( fcntl(a1, 6, &v10, a4, a5, a6) == -1 )
  {
    sub_2241E50(a1, 6, v6, v7, v8);
    return (unsigned int)*__errno_location();
  }
  else
  {
    sub_2241E40(a1, 6, v6, v7, v8);
    return 0;
  }
}
