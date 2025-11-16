// Function: sub_2795C70
// Address: 0x2795c70
//
__int64 __fastcall sub_2795C70(__int64 a1, int a2, unsigned int a3, __int64 a4, __int64 a5)
{
  unsigned int v5; // r12d
  _BYTE v7[16]; // [rsp+0h] [rbp-50h] BYREF
  char *v8; // [rsp+10h] [rbp-40h]
  char v9; // [rsp+20h] [rbp-30h] BYREF

  sub_27940D0((__int64)v7, a1, a2, a3, a4, a5);
  v5 = sub_2792D30(a1, (__int64)v7);
  if ( v8 != &v9 )
    _libc_free((unsigned __int64)v8);
  return v5;
}
