// Function: sub_E3D0C0
// Address: 0xe3d0c0
//
__int64 __fastcall sub_E3D0C0(_QWORD *a1, __int64 a2)
{
  __int64 v2; // rsi
  __int64 v4[2]; // [rsp+0h] [rbp-90h] BYREF
  __int64 v5; // [rsp+10h] [rbp-80h]
  __int64 v6; // [rsp+18h] [rbp-78h]
  unsigned int v7; // [rsp+20h] [rbp-70h]
  _BYTE *v8; // [rsp+28h] [rbp-68h]
  __int64 v9; // [rsp+30h] [rbp-60h]
  _BYTE v10[80]; // [rsp+38h] [rbp-58h] BYREF

  *a1 = a2;
  v2 = *(_QWORD *)(a2 + 80);
  v9 = 0x800000000LL;
  v4[0] = (__int64)a1;
  if ( v2 )
    v2 -= 24;
  v5 = 0;
  v4[1] = 0;
  v6 = 0;
  v7 = 0;
  v8 = v10;
  sub_E3C3B0(v4, v2);
  if ( v8 != v10 )
    _libc_free(v8, v2);
  return sub_C7D6A0(v5, 16LL * v7, 8);
}
