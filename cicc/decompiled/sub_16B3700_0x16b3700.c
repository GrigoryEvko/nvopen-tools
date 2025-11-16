// Function: sub_16B3700
// Address: 0x16b3700
//
__int64 __fastcall sub_16B3700(__int64 a1, int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  int v7; // edx
  _DWORD v8[5]; // [rsp+14h] [rbp-14h] BYREF

  v8[0] = 0;
  result = sub_16B3650(a1 + 184, a1, a3, a4, a5, a6, v8);
  if ( !(_BYTE)result )
  {
    v7 = v8[0];
    *(_DWORD *)(a1 + 16) = a2;
    *(_DWORD *)(a1 + 160) = v7;
  }
  return result;
}
