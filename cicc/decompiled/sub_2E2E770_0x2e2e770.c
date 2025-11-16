// Function: sub_2E2E770
// Address: 0x2e2e770
//
__int64 __fastcall sub_2E2E770(__int64 a1, __int64 a2)
{
  unsigned int v2; // r12d
  unsigned __int64 v4[2]; // [rsp+0h] [rbp-A0h] BYREF
  _BYTE v5[144]; // [rsp+10h] [rbp-90h] BYREF

  v4[0] = (unsigned __int64)v5;
  v4[1] = 0x1000000000LL;
  v2 = sub_2E2E630((__int64)v4, a2);
  if ( (_BYTE *)v4[0] != v5 )
    _libc_free(v4[0]);
  return v2;
}
