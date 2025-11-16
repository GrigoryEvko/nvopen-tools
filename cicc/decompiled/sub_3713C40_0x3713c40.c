// Function: sub_3713C40
// Address: 0x3713c40
//
unsigned __int64 *__fastcall sub_3713C40(unsigned __int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  _QWORD *v4; // r12
  unsigned __int64 v6; // rax
  __int64 v7; // r8
  __int64 v8; // r9
  unsigned __int64 v10; // [rsp+8h] [rbp-A8h] BYREF
  __int64 v11[2]; // [rsp+10h] [rbp-A0h] BYREF
  __int64 v12; // [rsp+20h] [rbp-90h] BYREF
  __int64 v13[2]; // [rsp+30h] [rbp-80h] BYREF
  __int64 v14; // [rsp+40h] [rbp-70h] BYREF
  __m128i v15[2]; // [rsp+50h] [rbp-60h] BYREF
  __int16 v16; // [rsp+70h] [rbp-40h]

  v4 = (_QWORD *)(a2 + 16);
  sub_37128E0(v11, (_QWORD *)(a2 + 16), *(_BYTE *)(a4 + 2) & 3, 0, 0);
  sub_8FD6D0((__int64)v13, "Attrs: ", v11);
  v16 = 260;
  v15[0].m128i_i64[0] = (__int64)v13;
  sub_370BC10(&v10, (_QWORD *)(a2 + 16), (unsigned __int16 *)(a4 + 2), v15);
  if ( (__int64 *)v13[0] != &v14 )
    j_j___libc_free_0(v13[0]);
  if ( (v10 & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    *a1 = v10 & 0xFFFFFFFFFFFFFFFELL | 1;
  }
  else
  {
    v15[0].m128i_i64[0] = (__int64)"BaseType";
    v16 = 259;
    sub_37011E0((unsigned __int64 *)v13, v4, (unsigned int *)(a4 + 4), v15[0].m128i_i64);
    v6 = v13[0] & 0xFFFFFFFFFFFFFFFELL;
    if ( (v13[0] & 0xFFFFFFFFFFFFFFFELL) != 0
      || (v13[0] = 0,
          sub_9C66B0(v13),
          v15[0].m128i_i64[0] = (__int64)"BaseOffset",
          v16 = 259,
          sub_3702660((unsigned __int64 *)v13, v4, (unsigned __int64 *)(a4 + 8), v15, v7, v8),
          v6 = v13[0] & 0xFFFFFFFFFFFFFFFELL,
          (v13[0] & 0xFFFFFFFFFFFFFFFELL) != 0) )
    {
      v13[0] = 0;
      *a1 = v6 | 1;
      sub_9C66B0(v13);
    }
    else
    {
      v13[0] = 0;
      sub_9C66B0(v13);
      *a1 = 1;
      v15[0].m128i_i64[0] = 0;
      sub_9C66B0(v15[0].m128i_i64);
    }
  }
  if ( (__int64 *)v11[0] != &v12 )
    j_j___libc_free_0(v11[0]);
  return a1;
}
