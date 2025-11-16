// Function: sub_154E850
// Address: 0x154e850
//
void __fastcall sub_154E850(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // r12
  _BYTE *v3; // rbx
  __int64 v4; // rsi
  _BYTE *v5; // [rsp+0h] [rbp-70h] BYREF
  __int64 v6; // [rsp+8h] [rbp-68h]
  _BYTE v7[96]; // [rsp+10h] [rbp-60h] BYREF

  v5 = v7;
  v6 = 0x400000000LL;
  sub_1626D60(a2, &v5);
  v2 = (unsigned __int64)v5;
  v3 = &v5[16 * (unsigned int)v6];
  if ( v3 != v5 )
  {
    do
    {
      v4 = *(_QWORD *)(v2 + 8);
      v2 += 16LL;
      sub_154E670(a1, v4);
    }
    while ( (_BYTE *)v2 != v3 );
    v2 = (unsigned __int64)v5;
  }
  if ( (_BYTE *)v2 != v7 )
    _libc_free(v2);
}
