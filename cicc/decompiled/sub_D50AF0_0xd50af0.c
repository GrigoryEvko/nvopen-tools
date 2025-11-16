// Function: sub_D50AF0
// Address: 0xd50af0
//
__int64 __fastcall sub_D50AF0(__int64 a1)
{
  __int64 *v1; // rbx
  __int64 *v2; // r12
  __int64 v3; // rdi
  __int64 v4; // rdi
  __int64 v5; // rsi
  __int64 v7; // [rsp+0h] [rbp-40h] BYREF
  __int64 v8; // [rsp+8h] [rbp-38h]
  __int64 v9; // [rsp+10h] [rbp-30h]
  __int64 v10; // [rsp+18h] [rbp-28h]

  v1 = *(__int64 **)(a1 + 32);
  v2 = *(__int64 **)(a1 + 40);
  v7 = 0;
  v8 = 0;
  v9 = 0;
  v10 = 0;
  if ( v1 == v2 )
  {
    v4 = 0;
    v5 = 0;
  }
  else
  {
    do
    {
      v3 = *v1++;
      sub_D50980(v3, (__int64)&v7);
    }
    while ( v2 != v1 );
    v4 = v8;
    v5 = 8LL * (unsigned int)v10;
  }
  return sub_C7D6A0(v4, v5, 8);
}
