// Function: sub_2D06DA0
// Address: 0x2d06da0
//
__int64 __fastcall sub_2D06DA0(__int64 a1, __int64 a2)
{
  __int64 v2; // rsi
  _BYTE v4[8]; // [rsp+0h] [rbp-6F0h] BYREF
  unsigned __int64 v5; // [rsp+8h] [rbp-6E8h]
  char v6; // [rsp+1Ch] [rbp-6D4h]
  char *v7; // [rsp+60h] [rbp-690h]
  char v8; // [rsp+70h] [rbp-680h] BYREF
  unsigned __int64 v9[54]; // [rsp+1B0h] [rbp-540h] BYREF
  _BYTE v10[8]; // [rsp+360h] [rbp-390h] BYREF
  unsigned __int64 v11; // [rsp+368h] [rbp-388h]
  char v12; // [rsp+37Ch] [rbp-374h]
  char *v13; // [rsp+3C0h] [rbp-330h]
  char v14; // [rsp+3D0h] [rbp-320h] BYREF
  _BYTE v15[8]; // [rsp+510h] [rbp-1E0h] BYREF
  unsigned __int64 v16; // [rsp+518h] [rbp-1D8h]
  char v17; // [rsp+52Ch] [rbp-1C4h]
  char *v18; // [rsp+570h] [rbp-180h]
  char v19; // [rsp+580h] [rbp-170h] BYREF

  v2 = *(_QWORD *)(a2 + 80);
  memset(v9, 0, sizeof(v9));
  LODWORD(v9[2]) = 8;
  v9[1] = (unsigned __int64)&v9[4];
  if ( v2 )
    v2 -= 24;
  BYTE4(v9[3]) = 1;
  v9[12] = (unsigned __int64)&v9[14];
  HIDWORD(v9[13]) = 8;
  sub_CE3280((__int64)v4, v2);
  sub_CE35F0((__int64)v15, (__int64)v9);
  sub_CE35F0((__int64)v10, (__int64)v4);
  sub_CE35F0(a1, (__int64)v10);
  sub_CE35F0(a1 + 432, (__int64)v15);
  if ( v13 != &v14 )
    _libc_free((unsigned __int64)v13);
  if ( !v12 )
    _libc_free(v11);
  if ( v18 != &v19 )
    _libc_free((unsigned __int64)v18);
  if ( !v17 )
    _libc_free(v16);
  if ( v7 != &v8 )
    _libc_free((unsigned __int64)v7);
  if ( !v6 )
    _libc_free(v5);
  if ( (unsigned __int64 *)v9[12] != &v9[14] )
    _libc_free(v9[12]);
  if ( !BYTE4(v9[3]) )
    _libc_free(v9[1]);
  return a1;
}
