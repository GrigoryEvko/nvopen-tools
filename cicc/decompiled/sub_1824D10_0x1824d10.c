// Function: sub_1824D10
// Address: 0x1824d10
//
_QWORD *__fastcall sub_1824D10(__int64 *a1, char *a2, signed __int64 a3)
{
  __int64 *v3; // rax
  __int64 v4; // r14
  __int64 v5; // rbx
  _QWORD *v6; // r12
  _BYTE v8[16]; // [rsp+0h] [rbp-40h] BYREF
  __int16 v9; // [rsp+10h] [rbp-30h]

  v3 = (__int64 *)sub_15996B0(*a1, a2, a3, 1);
  v4 = *v3;
  v5 = (__int64)v3;
  v9 = 257;
  v6 = sub_1648A60(88, 1u);
  if ( v6 )
    sub_15E51E0((__int64)v6, (__int64)a1, v4, 1, 8, v5, (__int64)v8, 0, 0, 0, 0);
  *((_BYTE *)v6 + 32) = v6[4] & 0x3F | 0x80;
  sub_15E4CC0((__int64)v6, 1u);
  return v6;
}
