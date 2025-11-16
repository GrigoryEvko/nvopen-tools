// Function: sub_26F9080
// Address: 0x26f9080
//
_BYTE *__fastcall sub_26F9080(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        unsigned __int64 a4,
        unsigned __int64 *a5,
        __int64 a6,
        unsigned __int8 *a7,
        size_t a8)
{
  _BYTE *v8; // r12
  char v9; // al
  __int64 v11[2]; // [rsp+0h] [rbp-30h] BYREF
  __int64 v12; // [rsp+10h] [rbp-20h] BYREF

  sub_26F78E0(v11, a3, a4, a5, a6, a6, a7, a8);
  v8 = sub_BA8D60(a1, v11[0], v11[1], a2);
  if ( (__int64 *)v11[0] != &v12 )
    j_j___libc_free_0(v11[0]);
  if ( *v8 == 3 )
  {
    v9 = v8[32] & 0xCF | 0x10;
    v8[32] = v9;
    if ( (v9 & 0xF) != 9 )
      v8[33] |= 0x40u;
  }
  return v8;
}
