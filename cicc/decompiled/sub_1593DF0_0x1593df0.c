// Function: sub_1593DF0
// Address: 0x1593df0
//
__int64 __fastcall sub_1593DF0(__int64 a1, __int64 a2, __int64 a3, _QWORD *a4)
{
  unsigned int v4; // eax
  unsigned int v5; // r12d
  __int64 v7; // [rsp+0h] [rbp-60h] BYREF
  _BYTE *v8; // [rsp+8h] [rbp-58h]
  _BYTE *v9; // [rsp+10h] [rbp-50h]
  __int64 v10; // [rsp+18h] [rbp-48h]
  int v11; // [rsp+20h] [rbp-40h]
  _BYTE v12[48]; // [rsp+28h] [rbp-38h] BYREF

  v7 = 0;
  v8 = v12;
  v9 = v12;
  v10 = 4;
  v11 = 0;
  LOBYTE(v4) = sub_1593C70(a1, &v7, a3, a4);
  v5 = v4;
  if ( v9 != v8 )
    _libc_free((unsigned __int64)v9);
  return v5;
}
