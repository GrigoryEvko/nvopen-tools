// Function: sub_1348400
// Address: 0x1348400
//
__int64 __fastcall sub_1348400(
        _BYTE *a1,
        __int64 a2,
        unsigned __int64 a3,
        unsigned __int64 a4,
        char a5,
        __int64 a6,
        __int64 a7,
        bool *a8)
{
  __int64 v8; // rbp
  __int64 result; // rax
  _QWORD v10[2]; // [rsp-10h] [rbp-10h] BYREF

  if ( a4 > 0x1000 || a5 )
    return 0;
  v10[1] = v8;
  result = 0;
  v10[0] = 0;
  if ( a3 <= *(_QWORD *)(a2 + 5624) )
  {
    sub_13481F0(a1, a2, a3, 1, (__int64)v10, a8);
    return v10[0];
  }
  return result;
}
