// Function: sub_2E25B90
// Address: 0x2e25b90
//
void __fastcall sub_2E25B90(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4)
{
  int i; // eax
  __int64 v6; // rcx
  _BYTE *v7; // [rsp+0h] [rbp-C0h] BYREF
  __int64 v8; // [rsp+8h] [rbp-B8h]
  _BYTE v9[176]; // [rsp+10h] [rbp-B0h] BYREF

  v8 = 0x1000000000LL;
  v7 = v9;
  sub_2E259B0(a1, a2, a3, a4, (__int64)&v7);
  for ( i = v8; (_DWORD)v8; i = v8 )
  {
    v6 = *(_QWORD *)&v7[8 * i - 8];
    LODWORD(v8) = i - 1;
    sub_2E259B0(a1, a2, a3, v6, (__int64)&v7);
  }
  if ( v7 != v9 )
    _libc_free((unsigned __int64)v7);
}
