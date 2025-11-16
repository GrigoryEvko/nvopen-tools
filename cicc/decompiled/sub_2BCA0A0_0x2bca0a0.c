// Function: sub_2BCA0A0
// Address: 0x2bca0a0
//
__int64 __fastcall sub_2BCA0A0(__int64 a1)
{
  __int64 *v1; // rax
  __int64 v2; // r12
  __int64 v4; // [rsp+0h] [rbp-40h] BYREF
  __int64 v5; // [rsp+8h] [rbp-38h]
  __int64 v6; // [rsp+10h] [rbp-30h] BYREF
  unsigned int v7; // [rsp+18h] [rbp-28h]
  char v8; // [rsp+30h] [rbp-10h] BYREF

  v1 = &v6;
  v4 = 0;
  v5 = 1;
  do
    *v1++ = -4096;
  while ( v1 != (__int64 *)&v8 );
  v2 = sub_2BC6BE0(a1, (__int64)&v4, 0, 0, 0);
  if ( (v5 & 1) == 0 )
    sub_C7D6A0(v6, 8LL * v7, 8);
  return v2;
}
