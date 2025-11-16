// Function: sub_31AA3D0
// Address: 0x31aa3d0
//
void __fastcall sub_31AA3D0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  __int64 *v7; // r13
  __int64 *v8; // rbx
  __int64 v9; // rsi
  __int64 v10; // [rsp+0h] [rbp-90h] BYREF
  char *v11; // [rsp+8h] [rbp-88h]
  __int64 v12; // [rsp+10h] [rbp-80h]
  int v13; // [rsp+18h] [rbp-78h]
  char v14; // [rsp+1Ch] [rbp-74h]
  char v15; // [rsp+20h] [rbp-70h] BYREF

  v11 = &v15;
  v6 = *a1;
  v10 = 0;
  v7 = *(__int64 **)(v6 + 40);
  v8 = *(__int64 **)(v6 + 32);
  v14 = 1;
  v12 = 8;
  v13 = 0;
  if ( v8 != v7 )
  {
    do
    {
      v9 = *v8++;
      sub_31A9620((__int64)a1, v9, &v10, (__int64)(a1 + 55), a5, a6);
    }
    while ( v7 != v8 );
    if ( !v14 )
      _libc_free((unsigned __int64)v11);
  }
}
