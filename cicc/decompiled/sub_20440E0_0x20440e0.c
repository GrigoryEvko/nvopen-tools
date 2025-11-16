// Function: sub_20440E0
// Address: 0x20440e0
//
__int64 __fastcall sub_20440E0(__int64 a1, int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  _DWORD v7[5]; // [rsp+14h] [rbp-14h] BYREF

  v7[0] = 0;
  result = sub_16B3650(a1 + 184, a1, a3, a4, a5, a6, v7);
  if ( !(_BYTE)result )
  {
    **(_DWORD **)(a1 + 160) = v7[0];
    *(_DWORD *)(a1 + 16) = a2;
  }
  return result;
}
