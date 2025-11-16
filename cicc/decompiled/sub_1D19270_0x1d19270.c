// Function: sub_1D19270
// Address: 0x1d19270
//
__int64 __fastcall sub_1D19270(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  unsigned int v6; // r12d
  unsigned __int64 v8[2]; // [rsp+0h] [rbp-1D0h] BYREF
  _QWORD v9[16]; // [rsp+10h] [rbp-1C0h] BYREF
  __int64 v10; // [rsp+90h] [rbp-140h] BYREF
  _BYTE *v11; // [rsp+98h] [rbp-138h]
  _BYTE *v12; // [rsp+A0h] [rbp-130h]
  __int64 v13; // [rsp+A8h] [rbp-128h]
  int v14; // [rsp+B0h] [rbp-120h]
  _BYTE v15[280]; // [rsp+B8h] [rbp-118h] BYREF

  v11 = v15;
  v12 = v15;
  v9[0] = a1;
  v10 = 0;
  v13 = 32;
  v14 = 0;
  v8[0] = (unsigned __int64)v9;
  v8[1] = 0x1000000001LL;
  v6 = sub_1D15B50(a2, (__int64)&v10, (__int64)v8, 0, 0, a6);
  if ( (_QWORD *)v8[0] != v9 )
    _libc_free(v8[0]);
  if ( v12 != v11 )
    _libc_free((unsigned __int64)v12);
  return v6;
}
