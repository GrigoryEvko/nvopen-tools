// Function: sub_F45E60
// Address: 0xf45e60
//
void __fastcall sub_F45E60(__int64 a1, __int64 a2, __int64 a3, unsigned int a4, __int64 a5, __int64 a6, __int64 a7)
{
  unsigned __int64 v9; // rsi
  _BYTE *v10; // r10
  _BYTE *v11; // r12
  __int64 v12; // r13
  __int64 v13; // r15
  unsigned int v14; // r14d
  _BYTE *v19; // [rsp+30h] [rbp-70h]
  __int64 v20; // [rsp+38h] [rbp-68h]
  _BYTE v21[8]; // [rsp+48h] [rbp-58h] BYREF
  _BYTE *v22; // [rsp+50h] [rbp-50h] BYREF
  __int64 v23; // [rsp+58h] [rbp-48h]
  _BYTE v24[64]; // [rsp+60h] [rbp-40h] BYREF

  v9 = (unsigned __int64)&v22;
  v22 = v24;
  v23 = 0x100000000LL;
  sub_B9A9D0(a2, (__int64)&v22);
  v10 = v22;
  v19 = &v22[16 * (unsigned int)v23];
  if ( v19 != v22 )
  {
    v11 = v22;
    v12 = a6;
    do
    {
      v13 = *((_QWORD *)v11 + 1);
      v11 += 16;
      v14 = *((_DWORD *)v11 - 4);
      sub_FC75A0(v21, a3, a4, a5, v12, a7);
      v20 = sub_FCD270(v21, v13);
      sub_FC7680(v21);
      v9 = v14;
      sub_B994D0(a1, v14, v20);
    }
    while ( v19 != v11 );
    v10 = v22;
  }
  if ( v10 != v24 )
    _libc_free(v10, v9);
}
