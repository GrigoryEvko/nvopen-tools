// Function: sub_2E0FDD0
// Address: 0x2e0fdd0
//
void __fastcall sub_2E0FDD0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, _QWORD *a6)
{
  __int64 v6; // rax
  __int128 *v7; // rbx
  __int128 *v8; // r13
  __int128 v10; // rax
  _QWORD v11[4]; // [rsp+20h] [rbp-1E0h] BYREF
  _BYTE *v12; // [rsp+40h] [rbp-1C0h]
  __int64 v13; // [rsp+48h] [rbp-1B8h]
  _BYTE v14[432]; // [rsp+50h] [rbp-1B0h] BYREF

  v13 = 0x1000000000LL;
  v6 = *(unsigned int *)(a2 + 8);
  v7 = *(__int128 **)a2;
  v11[0] = a1;
  v12 = v14;
  v11[1] = 0;
  v8 = (__int128 *)((char *)v7 + 24 * v6);
  while ( v8 != v7 )
  {
    v10 = *v7;
    v7 = (__int128 *)((char *)v7 + 24);
    sub_2E0F380((__int64)v11, a2, *((__int64 *)&v10 + 1), a4, a5, a6, v10, a3);
  }
  sub_2E0B930((__int64)v11, a2, a3, a4, a5);
  if ( v12 != v14 )
    _libc_free((unsigned __int64)v12);
}
