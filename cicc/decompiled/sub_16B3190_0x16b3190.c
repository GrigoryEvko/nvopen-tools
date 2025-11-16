// Function: sub_16B3190
// Address: 0x16b3190
//
__int64 __fastcall sub_16B3190(__int64 a1, int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  char v7; // dl
  _BYTE v8[17]; // [rsp+17h] [rbp-11h] BYREF

  v8[0] = 0;
  result = sub_16B3040(a1 + 184, a1, a3, a4, a5, a6, v8);
  if ( !(_BYTE)result )
  {
    v7 = v8[0];
    *(_DWORD *)(a1 + 16) = a2;
    *(_BYTE *)(a1 + 160) = v7;
  }
  return result;
}
