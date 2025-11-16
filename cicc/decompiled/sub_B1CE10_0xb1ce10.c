// Function: sub_B1CE10
// Address: 0xb1ce10
//
__int64 __fastcall sub_B1CE10(__int64 a1, __int64 a2)
{
  unsigned int v2; // r12d
  char *v4; // [rsp+0h] [rbp-60h] BYREF
  int v5; // [rsp+8h] [rbp-58h]
  char v6; // [rsp+10h] [rbp-50h] BYREF

  sub_B1CB80(&v4, a1, a2);
  LOBYTE(v2) = v5 != 0;
  if ( v4 != &v6 )
    _libc_free(v4, a1);
  return v2;
}
