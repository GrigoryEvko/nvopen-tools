// Function: sub_2664770
// Address: 0x2664770
//
void __fastcall sub_2664770(__int64 a1, __int64 a2, const void *a3, size_t a4)
{
  __int64 *v6; // rdi
  __int64 *v7; // r13
  __int64 *v8; // r15
  __int64 v9; // rcx
  __int64 *v10; // [rsp+10h] [rbp-60h] BYREF
  __int64 v11; // [rsp+18h] [rbp-58h]
  _BYTE v12[80]; // [rsp+20h] [rbp-50h] BYREF

  v10 = (__int64 *)v12;
  v11 = 0x400000000LL;
  sub_B91DD0(a1, a3, a4, (__int64)&v10);
  v6 = v10;
  v7 = &v10[(unsigned int)v11];
  if ( v7 != v10 )
  {
    v8 = v10;
    do
    {
      v9 = *v8++;
      sub_B99670(a2, a3, a4, v9);
    }
    while ( v7 != v8 );
    v6 = v10;
  }
  if ( v6 != (__int64 *)v12 )
    _libc_free((unsigned __int64)v6);
}
