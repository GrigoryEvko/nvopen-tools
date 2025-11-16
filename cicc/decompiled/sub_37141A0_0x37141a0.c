// Function: sub_37141A0
// Address: 0x37141a0
//
unsigned __int64 *__fastcall sub_37141A0(unsigned __int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  _QWORD *v4; // r12
  unsigned __int64 v6; // rax
  unsigned int v7; // r8d
  unsigned __int64 v9; // [rsp+8h] [rbp-A8h] BYREF
  __int64 v10[2]; // [rsp+10h] [rbp-A0h] BYREF
  __int64 v11; // [rsp+20h] [rbp-90h] BYREF
  __int64 v12[2]; // [rsp+30h] [rbp-80h] BYREF
  __int64 v13; // [rsp+40h] [rbp-70h] BYREF
  __m128i v14[2]; // [rsp+50h] [rbp-60h] BYREF
  __int16 v15; // [rsp+70h] [rbp-40h]

  v4 = (_QWORD *)(a2 + 16);
  sub_37128E0(v10, (_QWORD *)(a2 + 16), *(_BYTE *)(a4 + 2) & 3, 0, 0);
  sub_8FD6D0((__int64)v12, "Attrs: ", v10);
  v15 = 260;
  v14[0].m128i_i64[0] = (__int64)v12;
  sub_370BC10(&v9, (_QWORD *)(a2 + 16), (unsigned __int16 *)(a4 + 2), v14);
  if ( (__int64 *)v12[0] != &v13 )
    j_j___libc_free_0(v12[0]);
  if ( (v9 & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    *a1 = v9 & 0xFFFFFFFFFFFFFFFELL | 1;
  }
  else
  {
    v14[0].m128i_i64[0] = (__int64)"Type";
    v15 = 259;
    sub_37011E0((unsigned __int64 *)v12, v4, (unsigned int *)(a4 + 4), v14[0].m128i_i64);
    v6 = v12[0] & 0xFFFFFFFFFFFFFFFELL;
    if ( (v12[0] & 0xFFFFFFFFFFFFFFFELL) != 0
      || (v12[0] = 0,
          sub_9C66B0(v12),
          v14[0].m128i_i64[0] = (__int64)"Name",
          v15 = 259,
          sub_3701560((unsigned __int64 *)v12, v4, (_QWORD *)(a4 + 8), v14, v7),
          v6 = v12[0] & 0xFFFFFFFFFFFFFFFELL,
          (v12[0] & 0xFFFFFFFFFFFFFFFELL) != 0) )
    {
      v12[0] = 0;
      *a1 = v6 | 1;
      sub_9C66B0(v12);
    }
    else
    {
      v12[0] = 0;
      sub_9C66B0(v12);
      *a1 = 1;
      v14[0].m128i_i64[0] = 0;
      sub_9C66B0(v14[0].m128i_i64);
    }
  }
  if ( (__int64 *)v10[0] != &v11 )
    j_j___libc_free_0(v10[0]);
  return a1;
}
