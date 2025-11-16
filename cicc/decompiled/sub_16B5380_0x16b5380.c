// Function: sub_16B5380
// Address: 0x16b5380
//
__int64 __fastcall sub_16B5380(__int64 a1, int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  _BYTE v7[17]; // [rsp+17h] [rbp-11h] BYREF

  v7[0] = 0;
  result = sub_16B3040(a1 + 176, a1, a3, a4, a5, a6, v7);
  if ( !(_BYTE)result )
  {
    result = v7[0];
    if ( v7[0] )
    {
      sub_16B4BC0(*(char **)(a1 + 160));
      exit(0);
    }
    *(_DWORD *)(a1 + 16) = a2;
  }
  return result;
}
