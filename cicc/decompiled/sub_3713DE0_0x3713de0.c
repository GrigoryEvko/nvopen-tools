// Function: sub_3713DE0
// Address: 0x3713de0
//
unsigned __int64 *__fastcall sub_3713DE0(unsigned __int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  _QWORD *v4; // r12
  __int64 v6; // r8
  __int64 v7; // r9
  unsigned __int64 v8; // rax
  unsigned int v9; // r8d
  unsigned __int64 v11; // [rsp+8h] [rbp-A8h] BYREF
  __int64 v12[2]; // [rsp+10h] [rbp-A0h] BYREF
  __int64 v13; // [rsp+20h] [rbp-90h] BYREF
  __int64 v14[2]; // [rsp+30h] [rbp-80h] BYREF
  __int64 v15; // [rsp+40h] [rbp-70h] BYREF
  __m128i v16[2]; // [rsp+50h] [rbp-60h] BYREF
  __int16 v17; // [rsp+70h] [rbp-40h]

  v4 = (_QWORD *)(a2 + 16);
  sub_37128E0(v12, (_QWORD *)(a2 + 16), *(_BYTE *)(a4 + 2) & 3, 0, 0);
  sub_8FD6D0((__int64)v14, "Attrs: ", v12);
  v17 = 260;
  v16[0].m128i_i64[0] = (__int64)v14;
  sub_370BC10(&v11, (_QWORD *)(a2 + 16), (unsigned __int16 *)(a4 + 2), v16);
  if ( (__int64 *)v14[0] != &v15 )
    j_j___libc_free_0(v14[0]);
  if ( (v11 & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    *a1 = v11 & 0xFFFFFFFFFFFFFFFELL | 1;
  }
  else
  {
    v16[0].m128i_i64[0] = (__int64)"EnumValue";
    v17 = 259;
    sub_3702790((unsigned __int64 *)v14, v4, a4 + 8, (__int64)v16, v6, v7);
    v8 = v14[0] & 0xFFFFFFFFFFFFFFFELL;
    if ( (v14[0] & 0xFFFFFFFFFFFFFFFELL) != 0
      || (v14[0] = 0,
          sub_9C66B0(v14),
          v16[0].m128i_i64[0] = (__int64)"Name",
          v17 = 259,
          sub_3701560((unsigned __int64 *)v14, v4, (_QWORD *)(a4 + 24), v16, v9),
          v8 = v14[0] & 0xFFFFFFFFFFFFFFFELL,
          (v14[0] & 0xFFFFFFFFFFFFFFFELL) != 0) )
    {
      v14[0] = 0;
      *a1 = v8 | 1;
      sub_9C66B0(v14);
    }
    else
    {
      v14[0] = 0;
      sub_9C66B0(v14);
      *a1 = 1;
      v16[0].m128i_i64[0] = 0;
      sub_9C66B0(v16[0].m128i_i64);
    }
  }
  if ( (__int64 *)v12[0] != &v13 )
    j_j___libc_free_0(v12[0]);
  return a1;
}
