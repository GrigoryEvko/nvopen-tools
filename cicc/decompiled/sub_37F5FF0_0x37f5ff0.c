// Function: sub_37F5FF0
// Address: 0x37f5ff0
//
void __fastcall sub_37F5FF0(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // [rsp+0h] [rbp-30h] BYREF
  char *v7; // [rsp+8h] [rbp-28h]
  __int64 v8; // [rsp+10h] [rbp-20h]
  int v9; // [rsp+18h] [rbp-18h]
  char v10; // [rsp+1Ch] [rbp-14h]
  char v11; // [rsp+20h] [rbp-10h] BYREF

  v6 = 0;
  v7 = &v11;
  v8 = 2;
  v9 = 0;
  v10 = 1;
  sub_37F5D50(a1, a2, a3, a4, (__int64)&v6, a6);
  if ( !v10 )
    _libc_free((unsigned __int64)v7);
}
