// Function: sub_1AEB370
// Address: 0x1aeb370
//
__int64 __fastcall sub_1AEB370(__int64 a1, __int64 *a2)
{
  __int64 result; // rax
  unsigned __int8 v3; // [rsp-B9h] [rbp-B9h]
  unsigned __int64 v4[2]; // [rsp-B8h] [rbp-B8h] BYREF
  _QWORD v5[21]; // [rsp-A8h] [rbp-A8h] BYREF

  if ( *(_BYTE *)(a1 + 16) <= 0x17u )
    return 0;
  if ( *(_QWORD *)(a1 + 8) )
    return 0;
  v3 = sub_1AE9990(a1, a2);
  if ( !v3 )
    return 0;
  v5[0] = a1;
  v4[1] = 0x1000000001LL;
  v4[0] = (unsigned __int64)v5;
  sub_1AEB210((__int64)v4, a2);
  result = v3;
  if ( (_QWORD *)v4[0] != v5 )
  {
    _libc_free(v4[0]);
    return v3;
  }
  return result;
}
