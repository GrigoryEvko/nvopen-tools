// Function: sub_1694AB0
// Address: 0x1694ab0
//
_QWORD *__fastcall sub_1694AB0(__int64 *a1, int a2, char *a3, unsigned __int64 a4)
{
  __int64 *v7; // r14
  _QWORD *v8; // r12
  char v9; // al
  __int64 v11; // [rsp+0h] [rbp-80h]
  __int64 *v12; // [rsp+10h] [rbp-70h] BYREF
  __int16 v13; // [rsp+20h] [rbp-60h]
  __int64 v14[2]; // [rsp+30h] [rbp-50h] BYREF
  __int64 v15; // [rsp+40h] [rbp-40h] BYREF

  switch ( a2 )
  {
    case 9:
      a2 = 2;
      break;
    case 1:
      a2 = 3;
      break;
    case 7:
    case 0:
      a2 = 8;
      break;
  }
  v7 = (__int64 *)sub_15996B0(*a1, a3, a4, 0);
  v11 = *v7;
  sub_16949F0(v14, (__int64)a3, a4, a2);
  v13 = 260;
  v12 = v14;
  v8 = sub_1648A60(88, 1u);
  if ( v8 )
    sub_15E51E0((__int64)v8, (__int64)a1, v11, 1, a2, (__int64)v7, (__int64)&v12, 0, 0, 0, 0);
  if ( (__int64 *)v14[0] != &v15 )
    j_j___libc_free_0(v14[0], v15 + 1);
  v9 = *((_BYTE *)v8 + 32);
  if ( (v9 & 0xFu) - 7 > 1 )
  {
    *((_BYTE *)v8 + 32) = v9 & 0xCF | 0x10;
    if ( (v9 & 0xF) != 9 )
      *((_BYTE *)v8 + 33) |= 0x40u;
  }
  return v8;
}
