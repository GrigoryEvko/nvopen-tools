// Function: sub_A7B2C0
// Address: 0xa7b2c0
//
unsigned __int64 __fastcall sub_A7B2C0(__int64 *a1, __int64 *a2, int a3, __int64 a4)
{
  __int64 v7; // rax
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // r12
  int v10; // [rsp+0h] [rbp-90h] BYREF
  char *v11; // [rsp+8h] [rbp-88h]
  char v12; // [rsp+18h] [rbp-78h] BYREF

  if ( !*(_DWORD *)(a4 + 16) )
    return *a1;
  if ( *a1 )
  {
    v7 = sub_A74490(a1, a3);
    sub_A74940((__int64)&v10, (__int64)a2, v7);
    sub_A776F0((__int64)&v10, a4);
    v8 = sub_A7A280(a2, (__int64)&v10);
    v9 = sub_A78500(a1, (unsigned __int64 *)a2, a3, v8);
    if ( v11 != &v12 )
      _libc_free(v11, a2);
    return v9;
  }
  else
  {
    v10 = a3;
    v11 = (char *)sub_A7A280(a2, a4);
    return sub_A78010(a2, &v10, 1);
  }
}
