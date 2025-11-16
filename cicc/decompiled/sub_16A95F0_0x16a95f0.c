// Function: sub_16A95F0
// Address: 0x16a95f0
//
void __fastcall sub_16A95F0(__int64 a1, __int64 a2, char a3)
{
  const char *v3; // [rsp+0h] [rbp-50h] BYREF
  __int64 v4; // [rsp+8h] [rbp-48h]
  _BYTE v5[64]; // [rsp+10h] [rbp-40h] BYREF

  v4 = 0x2800000000LL;
  v3 = v5;
  sub_16A8FA0(a1, (__int64)&v3, 0xAu, a3, 0);
  sub_16E7EE0(a2, v3, (unsigned int)v4);
  if ( v3 != v5 )
    _libc_free((unsigned __int64)v3);
}
