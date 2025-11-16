// Function: sub_24E3D70
// Address: 0x24e3d70
//
void __fastcall sub_24E3D70(__int64 *a1, __int64 *a2, int a3, __int64 a4, char a5, char a6)
{
  unsigned __int16 v8; // ax
  unsigned __int64 v9; // rax
  _BYTE *v10; // rdi
  __int64 *v14; // [rsp+10h] [rbp-90h] BYREF
  _BYTE *v15; // [rsp+18h] [rbp-88h]
  __int64 v16; // [rsp+20h] [rbp-80h]
  _BYTE v17[120]; // [rsp+28h] [rbp-78h] BYREF

  v14 = a2;
  v15 = v17;
  v16 = 0x800000000LL;
  sub_A77B20(&v14, 43);
  sub_A77B20(&v14, 40);
  if ( a6 )
    sub_A77B20(&v14, 22);
  HIBYTE(v8) = 1;
  LOBYTE(v8) = a5;
  sub_A77B90(&v14, v8);
  sub_A77BF0(&v14, a4);
  v9 = sub_A7B2C0(a1, a2, a3 + 1, (__int64)&v14);
  v10 = v15;
  *a1 = v9;
  if ( v10 != v17 )
    _libc_free((unsigned __int64)v10);
}
