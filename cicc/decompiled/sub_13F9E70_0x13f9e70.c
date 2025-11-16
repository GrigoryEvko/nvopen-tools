// Function: sub_13F9E70
// Address: 0x13f9e70
//
__int64 __fastcall sub_13F9E70(__int64 a1)
{
  __int64 v1; // r12
  _BYTE *v3; // [rsp+0h] [rbp-60h] BYREF
  __int64 v4; // [rsp+8h] [rbp-58h]
  _BYTE v5[80]; // [rsp+10h] [rbp-50h] BYREF

  v1 = 0;
  v3 = v5;
  v4 = 0x800000000LL;
  sub_13F9CA0(a1, (__int64)&v3);
  if ( (_DWORD)v4 == 1 )
    v1 = *(_QWORD *)v3;
  if ( v3 != v5 )
    _libc_free((unsigned __int64)v3);
  return v1;
}
