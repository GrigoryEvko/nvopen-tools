// Function: sub_12658E0
// Address: 0x12658e0
//
__int64 __fastcall sub_12658E0(
        __int64 a1,
        __int64 *a2,
        __int64 a3,
        _DWORD *a4,
        char a5,
        unsigned __int8 a6,
        char a7,
        char a8)
{
  int v8; // ebx
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  _BYTE v17[56]; // [rsp+28h] [rbp-38h] BYREF

  v8 = a6;
  sub_1602D10(v17);
  sub_1265340(a1, (size_t)v17, a2, a3, a4, a5, v8, a7, a8);
  return sub_16025D0(v17, v17, v9, v10, v11, v12);
}
