// Function: sub_905E50
// Address: 0x905e50
//
__int64 __fastcall sub_905E50(
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
  _BYTE v13[56]; // [rsp+28h] [rbp-38h] BYREF

  v8 = a6;
  sub_B6EEA0(v13);
  sub_905880(a1, (__int64)v13, a2, a3, a4, a5, v8, a7, a8);
  return sub_B6E710(v13);
}
