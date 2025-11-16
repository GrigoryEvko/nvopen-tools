// Function: sub_33CFFC0
// Address: 0x33cffc0
//
__int64 __fastcall sub_33CFFC0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // r12d
  unsigned __int64 v8[2]; // [rsp+0h] [rbp-1C0h] BYREF
  _QWORD v9[16]; // [rsp+10h] [rbp-1B0h] BYREF
  __int64 v10; // [rsp+90h] [rbp-130h] BYREF
  char *v11; // [rsp+98h] [rbp-128h]
  __int64 v12; // [rsp+A0h] [rbp-120h]
  int v13; // [rsp+A8h] [rbp-118h]
  char v14; // [rsp+ACh] [rbp-114h]
  char v15; // [rsp+B0h] [rbp-110h] BYREF

  v11 = &v15;
  v9[0] = a1;
  v10 = 0;
  v12 = 32;
  v13 = 0;
  v14 = 1;
  v8[0] = (unsigned __int64)v9;
  v8[1] = 0x1000000001LL;
  v6 = sub_3285B00(a2, (__int64)&v10, (__int64)v8, 0, 0, a6);
  if ( (_QWORD *)v8[0] != v9 )
    _libc_free(v8[0]);
  if ( !v14 )
    _libc_free((unsigned __int64)v11);
  return v6;
}
