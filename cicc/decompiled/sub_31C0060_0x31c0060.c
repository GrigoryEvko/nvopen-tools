// Function: sub_31C0060
// Address: 0x31c0060
//
void __fastcall sub_31C0060(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        char a7,
        __int64 a8)
{
  __int64 v8; // rbx
  __int64 v9; // r13
  __int64 v10; // rax
  _QWORD *v11; // r12
  _BYTE v12[8]; // [rsp+0h] [rbp-40h] BYREF
  __int64 v13; // [rsp+8h] [rbp-38h]

  v8 = *a1;
  v9 = *a1 + 8LL * *((unsigned int *)a1 + 2);
  if ( v9 != *a1 )
  {
    do
    {
      v11 = *(_QWORD **)(*(_QWORD *)v8 + 8LL);
      sub_318B480((__int64)v12, (__int64)v11);
      if ( v13 == a8 )
        sub_371B2F0(&a7);
      v8 += 8;
      v10 = sub_371B390(&a7);
      sub_318CB50(v11, v10, (__int64)&a7);
    }
    while ( v9 != v8 );
  }
}
