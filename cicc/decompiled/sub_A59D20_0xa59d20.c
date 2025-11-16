// Function: sub_A59D20
// Address: 0xa59d20
//
void __fastcall sub_A59D20(__int64 a1, __int64 a2)
{
  _BYTE *v4; // rsi
  _BYTE *v5; // r12
  _BYTE *v6; // rbx
  _BYTE *v7; // [rsp+0h] [rbp-70h] BYREF
  __int64 v8; // [rsp+8h] [rbp-68h]
  _BYTE v9[96]; // [rsp+10h] [rbp-60h] BYREF

  v4 = &v7;
  v7 = v9;
  v8 = 0x400000000LL;
  sub_B9A9D0(a2, &v7);
  v5 = v7;
  v6 = &v7[16 * (unsigned int)v8];
  if ( v6 != v7 )
  {
    do
    {
      v4 = (_BYTE *)*((_QWORD *)v5 + 1);
      v5 += 16;
      sub_A59AF0(a1, v4);
    }
    while ( v5 != v6 );
    v5 = v7;
  }
  if ( v5 != v9 )
    _libc_free(v5, v4);
}
