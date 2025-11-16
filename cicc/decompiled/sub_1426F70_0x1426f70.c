// Function: sub_1426F70
// Address: 0x1426f70
//
void __fastcall sub_1426F70(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 *v4; // rdi
  __int64 *v5; // r13
  __int64 *v6; // rbx
  __int64 v7; // rsi
  __int64 v8; // [rsp+0h] [rbp-150h] BYREF
  char v9; // [rsp+8h] [rbp-148h]
  __int64 v10; // [rsp+18h] [rbp-138h]
  __int64 *v11; // [rsp+20h] [rbp-130h] BYREF
  __int64 v12; // [rsp+28h] [rbp-128h]
  _BYTE v13[288]; // [rsp+30h] [rbp-120h] BYREF

  v3 = *(_QWORD *)(a1 + 8);
  v10 = a2;
  v9 = 0;
  v8 = v3;
  v12 = 0x2000000000LL;
  v11 = (__int64 *)v13;
  sub_14DECB0(&v8, &v11);
  v4 = v11;
  v5 = &v11[(unsigned int)v12];
  if ( v5 != v11 )
  {
    v6 = v11;
    do
    {
      v7 = *v6++;
      sub_1426C80(a1, v7);
    }
    while ( v5 != v6 );
    v4 = v11;
  }
  if ( v4 != (__int64 *)v13 )
    _libc_free((unsigned __int64)v4);
}
