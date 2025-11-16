// Function: sub_13F9B10
// Address: 0x13f9b10
//
__int64 __fastcall sub_13F9B10(__int64 a1, int a2, int a3, int a4, int a5, int a6)
{
  __int64 result; // rax
  _BYTE v7[17]; // [rsp+17h] [rbp-11h] BYREF

  v7[0] = 0;
  result = sub_16B3040((int)a1 + 184, a1, a3, a4, a5, a6, (__int64)v7);
  if ( !(_BYTE)result )
  {
    **(_BYTE **)(a1 + 160) = v7[0];
    *(_DWORD *)(a1 + 16) = a2;
  }
  return result;
}
