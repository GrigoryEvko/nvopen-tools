// Function: sub_2E43B30
// Address: 0x2e43b30
//
void __fastcall sub_2E43B30(__int64 a1, void **a2, char a3)
{
  __int64 *v4; // rdi
  __int64 v5; // [rsp+8h] [rbp-98h] BYREF
  __int64 *v6; // [rsp+10h] [rbp-90h] BYREF
  __int64 v7; // [rsp+18h] [rbp-88h]
  __int64 v8; // [rsp+20h] [rbp-80h] BYREF
  __int64 v9[2]; // [rsp+30h] [rbp-70h] BYREF
  _QWORD v10[2]; // [rsp+40h] [rbp-60h] BYREF
  void *v11; // [rsp+50h] [rbp-50h] BYREF
  __int16 v12; // [rsp+70h] [rbp-30h]

  v5 = a1;
  v12 = 257;
  v9[0] = (__int64)v10;
  sub_2E396D0(v9, byte_3F871B3, (__int64)byte_3F871B3);
  sub_2E43380((__int64)&v6, (size_t)&v5, a2, a3, &v11, (__int64)v9);
  if ( (_QWORD *)v9[0] != v10 )
    j_j___libc_free_0(v9[0]);
  v4 = v6;
  if ( v7 )
  {
    sub_C67930(v6, v7, 0, 0);
    v4 = v6;
  }
  if ( v4 != &v8 )
    j_j___libc_free_0((unsigned __int64)v4);
}
