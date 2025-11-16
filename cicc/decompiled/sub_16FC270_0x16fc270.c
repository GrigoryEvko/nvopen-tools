// Function: sub_16FC270
// Address: 0x16fc270
//
_QWORD *__fastcall sub_16FC270(unsigned __int64 **a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  _QWORD *result; // rax
  _BYTE v6[24]; // [rsp+0h] [rbp-40h] BYREF
  _QWORD *v7; // [rsp+18h] [rbp-28h]
  _QWORD v8[3]; // [rsp+28h] [rbp-18h] BYREF

  sub_16FC210((__int64)v6, a1, a3, a4, a5);
  result = v8;
  if ( v7 != v8 )
    return (_QWORD *)j_j___libc_free_0(v7, v8[0] + 1LL);
  return result;
}
