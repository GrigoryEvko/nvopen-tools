// Function: sub_253A4F0
// Address: 0x253a4f0
//
__int64 __fastcall sub_253A4F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v7[2]; // [rsp+8h] [rbp-28h] BYREF
  _BYTE v8[16]; // [rsp+18h] [rbp-18h] BYREF

  v7[0] = (unsigned __int64)v8;
  v7[1] = 0;
  if ( !*(_DWORD *)(a2 + 16) )
    return 0;
  sub_2538240((__int64)v7, (char **)(a2 + 8), a3, a4, a5, a6);
  if ( (_BYTE *)v7[0] == v8 )
    return 0;
  _libc_free(v7[0]);
  return 0;
}
