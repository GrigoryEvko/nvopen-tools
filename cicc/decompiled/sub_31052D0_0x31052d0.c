// Function: sub_31052D0
// Address: 0x31052d0
//
__int64 __fastcall sub_31052D0(__int64 a1, __int64 a2)
{
  unsigned int v2; // r13d
  char *v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // r9
  __int64 v8[2]; // [rsp+0h] [rbp-70h] BYREF
  _BYTE v9[96]; // [rsp+10h] [rbp-60h] BYREF

  v2 = 0;
  if ( a2 )
  {
    v8[0] = (__int64)v9;
    v8[1] = 0x800000000LL;
    sub_3104F60((__int64)v8, a1);
    v2 = sub_3103B70(v8, a2, v3, v4, v5, v6);
    if ( (_BYTE *)v8[0] != v9 )
      _libc_free(v8[0]);
  }
  return v2;
}
