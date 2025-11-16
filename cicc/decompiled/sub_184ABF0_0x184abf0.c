// Function: sub_184ABF0
// Address: 0x184abf0
//
__int64 __fastcall sub_184ABF0(__int64 a1, __int64 a2)
{
  unsigned __int64 *v2; // rax
  unsigned int v3; // r14d
  __int64 v5; // [rsp+0h] [rbp-C0h] BYREF
  __int64 v6; // [rsp+8h] [rbp-B8h]
  __int64 v7; // [rsp+10h] [rbp-B0h] BYREF
  unsigned __int64 v8[2]; // [rsp+50h] [rbp-70h] BYREF
  _BYTE v9[96]; // [rsp+60h] [rbp-60h] BYREF

  v2 = (unsigned __int64 *)&v7;
  v5 = 0;
  v6 = 1;
  do
    *v2++ = -8;
  while ( v2 != v8 );
  v3 = 0;
  v8[0] = (unsigned __int64)v9;
  v8[1] = 0x800000000LL;
  if ( (unsigned int)sub_134CE70(a2, a1) != 4 )
    v3 = sub_1849A60(a1, a2, (__int64)&v5);
  if ( (_BYTE *)v8[0] != v9 )
    _libc_free(v8[0]);
  if ( (v6 & 1) == 0 )
    j___libc_free_0(v7);
  return v3;
}
