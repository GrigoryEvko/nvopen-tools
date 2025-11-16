// Function: sub_C82790
// Address: 0xc82790
//
__int64 __fastcall sub_C82790(__int64 a1, bool *a2)
{
  __int64 result; // rax
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // r8
  _OWORD v6[2]; // [rsp+0h] [rbp-60h] BYREF
  __int128 v7; // [rsp+20h] [rbp-40h]
  __int128 v8; // [rsp+30h] [rbp-30h]
  __int64 v9; // [rsp+40h] [rbp-20h]

  v7 = 0;
  v9 = 0;
  HIDWORD(v7) = 0xFFFF;
  memset(v6, 0, sizeof(v6));
  v8 = 0;
  result = sub_C826E0(a1, (__int64)v6, 1);
  if ( !(_DWORD)result )
  {
    *a2 = sub_C81FB0((__int64)v6);
    sub_2241E40(v6, v6, v3, v4, v5);
    return 0;
  }
  return result;
}
