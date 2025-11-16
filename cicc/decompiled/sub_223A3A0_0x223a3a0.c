// Function: sub_223A3A0
// Address: 0x223a3a0
//
_QWORD *__fastcall sub_223A3A0(
        __int64 a1,
        _QWORD *a2,
        __int64 a3,
        _QWORD *a4,
        __int64 a5,
        __int64 a6,
        _DWORD *a7,
        unsigned __int64 *a8)
{
  int v9; // r12d
  _QWORD *result; // rax
  unsigned __int64 v11; // rsi
  unsigned __int64 v12[4]; // [rsp+18h] [rbp-20h] BYREF

  v9 = *(_DWORD *)(a6 + 24);
  *(_DWORD *)(a6 + 24) = v9 & 0xFFFFFFB5 | 8;
  result = sub_2239910(a1, a2, a3, a4, a5, a6, a7, v12);
  v11 = v12[0];
  *(_DWORD *)(a6 + 24) = v9;
  *a8 = v11;
  return result;
}
