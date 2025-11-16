// Function: sub_1403400
// Address: 0x1403400
//
__int64 __fastcall sub_1403400(__int64 a1)
{
  __int64 *v1; // rbx
  __int64 *v2; // r12
  __int64 v3; // rdi
  __int64 v5; // [rsp+0h] [rbp-40h] BYREF
  __int64 v6; // [rsp+8h] [rbp-38h]
  __int64 v7; // [rsp+10h] [rbp-30h]
  __int64 v8; // [rsp+18h] [rbp-28h]

  v1 = *(__int64 **)(a1 + 32);
  v2 = *(__int64 **)(a1 + 40);
  v5 = 0;
  v6 = 0;
  v7 = 0;
  v8 = 0;
  if ( v1 == v2 )
    return j___libc_free_0(0);
  do
  {
    v3 = *v1++;
    sub_14031A0(v3, (__int64)&v5);
  }
  while ( v2 != v1 );
  return j___libc_free_0(v6);
}
