// Function: sub_2CF2AC0
// Address: 0x2cf2ac0
//
__int64 __fastcall sub_2CF2AC0(_QWORD *a1, __int64 a2, __int64 a3)
{
  int v4; // r13d
  unsigned __int8 *v5; // r9
  _BYTE *v6; // r12
  unsigned __int8 v7; // bl
  __int64 v8; // rsi
  char v9; // al
  unsigned __int8 v11; // [rsp+17h] [rbp-C9h]
  _BYTE *v12; // [rsp+20h] [rbp-C0h]
  unsigned __int8 *v13; // [rsp+28h] [rbp-B8h]
  unsigned __int8 v14; // [rsp+3Fh] [rbp-A1h] BYREF
  __int64 v15; // [rsp+40h] [rbp-A0h] BYREF
  __int64 v16; // [rsp+48h] [rbp-98h]
  __int64 v17; // [rsp+50h] [rbp-90h]
  unsigned int v18; // [rsp+58h] [rbp-88h]
  _BYTE *v19; // [rsp+60h] [rbp-80h] BYREF
  __int64 v20; // [rsp+68h] [rbp-78h]
  _BYTE v21[112]; // [rsp+70h] [rbp-70h] BYREF

  v4 = 0;
  v19 = v21;
  v15 = 0;
  v16 = 0;
  v17 = 0;
  v18 = 0;
  v14 = 0;
  v20 = 0x800000000LL;
  sub_2CF2750((__int64)&v19, a2);
  v11 = 0;
  v5 = &v14;
  while ( 1 )
  {
    ++v4;
    v6 = &v19[8 * (unsigned int)v20];
    v12 = v19;
    if ( v19 == v6 )
      break;
    v7 = 0;
    do
    {
      v8 = *((_QWORD *)v6 - 1);
      v13 = v5;
      v6 -= 8;
      v9 = sub_2CEDAC0(a1, v8, v4, a3, (unsigned __int8 *)&v15, v5);
      v5 = v13;
      v7 |= v9;
    }
    while ( v12 != v6 );
    if ( !v7 )
    {
      v6 = v19;
      break;
    }
    v11 = v14;
    if ( !v14 )
    {
      v11 = v7;
      v6 = v19;
      break;
    }
  }
  if ( v6 != v21 )
    _libc_free((unsigned __int64)v6);
  sub_C7D6A0(v16, 16LL * v18, 8);
  return v11;
}
