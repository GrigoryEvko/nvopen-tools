// Function: sub_279FB50
// Address: 0x279fb50
//
__int64 __fastcall sub_279FB50(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // r13d
  _BYTE *v7; // r14
  _BYTE *v8; // r12
  __int64 v9; // rsi
  _BYTE *v11; // [rsp+0h] [rbp-80h] BYREF
  __int64 v12; // [rsp+8h] [rbp-78h]
  _BYTE v13[112]; // [rsp+10h] [rbp-70h] BYREF

  v6 = 0;
  sub_278F580(a1, a2, a3, a4, a5, a6);
  v11 = v13;
  v12 = 0x800000000LL;
  sub_2797FE0((__int64)&v11, a2);
  v7 = v11;
  v8 = &v11[8 * (unsigned int)v12];
  if ( v11 != v8 )
  {
    do
    {
      v9 = *((_QWORD *)v8 - 1);
      v8 -= 8;
      v6 |= sub_279F630(a1, v9);
    }
    while ( v7 != v8 );
    v8 = v11;
  }
  if ( v8 != v13 )
    _libc_free((unsigned __int64)v8);
  return v6;
}
