// Function: sub_1600640
// Address: 0x1600640
//
__int64 __fastcall sub_1600640(__int64 a1)
{
  __int64 v1; // rdx
  __int64 *v2; // rsi
  int v3; // edi
  _BYTE v5[16]; // [rsp+0h] [rbp-20h] BYREF
  __int16 v6; // [rsp+10h] [rbp-10h]

  v1 = *(_QWORD *)(a1 - 24);
  v2 = *(__int64 **)(a1 - 48);
  v3 = *(unsigned __int8 *)(a1 + 16);
  v6 = 257;
  return sub_15FB440(v3 - 24, v2, v1, (__int64)v5, 0);
}
