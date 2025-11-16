// Function: sub_2EB57C0
// Address: 0x2eb57c0
//
__int64 __fastcall sub_2EB57C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned int v4; // r12d
  char *v6; // [rsp+0h] [rbp-60h] BYREF
  int v7; // [rsp+8h] [rbp-58h]
  char v8; // [rsp+10h] [rbp-50h] BYREF

  sub_2EB5530(&v6, a1, a2, a4, a1);
  LOBYTE(v4) = v7 != 0;
  if ( v6 != &v8 )
    _libc_free((unsigned __int64)v6);
  return v4;
}
