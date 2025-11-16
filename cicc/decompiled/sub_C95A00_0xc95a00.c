// Function: sub_C95A00
// Address: 0xc95a00
//
__int64 __fastcall sub_C95A00(__int64 a1)
{
  __int64 result; // rax
  _QWORD v2[3]; // [rsp+0h] [rbp-70h] BYREF
  _BYTE v3[80]; // [rsp+18h] [rbp-58h] BYREF

  v2[0] = v3;
  v2[1] = 0;
  v2[2] = 64;
  sub_CA12A0(a1, v2);
  result = sub_C959F0();
  if ( (_DWORD)result )
    result = sub_C959F0();
  if ( (_BYTE *)v2[0] != v3 )
    return _libc_free(v2[0], v2);
  return result;
}
