// Function: sub_13BB990
// Address: 0x13bb990
//
_QWORD *__fastcall sub_13BB990(__int64 a1, __int64 a2, char a3, __int64 a4, unsigned int a5)
{
  _QWORD *result; // rax
  _QWORD *v7; // [rsp+0h] [rbp-30h] BYREF
  __int64 v8; // [rsp+8h] [rbp-28h]
  _QWORD v9[3]; // [rsp+10h] [rbp-20h] BYREF

  sub_13BB510((__int64)&v7, a1, a2, a3, a4);
  if ( v8 )
    sub_16BED90(v7, v8, 0, a5);
  result = v9;
  if ( v7 != v9 )
    return (_QWORD *)j_j___libc_free_0(v7, v9[0] + 1LL);
  return result;
}
