// Function: sub_3713F80
// Address: 0x3713f80
//
__int64 *__fastcall sub_3713F80(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  _QWORD *v4; // r12
  unsigned __int64 v6; // rax
  __int64 v7; // r8
  __int64 v8; // r9
  unsigned int v9; // r8d
  unsigned __int64 v10; // rax
  unsigned __int64 v12; // [rsp+8h] [rbp-A8h] BYREF
  __int64 v13[2]; // [rsp+10h] [rbp-A0h] BYREF
  __int64 v14; // [rsp+20h] [rbp-90h] BYREF
  __int64 v15[2]; // [rsp+30h] [rbp-80h] BYREF
  __int64 v16; // [rsp+40h] [rbp-70h] BYREF
  __m128i v17[2]; // [rsp+50h] [rbp-60h] BYREF
  __int16 v18; // [rsp+70h] [rbp-40h]

  v4 = (_QWORD *)(a2 + 16);
  sub_37128E0(v13, (_QWORD *)(a2 + 16), *(_BYTE *)(a4 + 2) & 3, 0, 0);
  sub_8FD6D0((__int64)v15, "Attrs: ", v13);
  v18 = 260;
  v17[0].m128i_i64[0] = (__int64)v15;
  sub_370BC10(&v12, (_QWORD *)(a2 + 16), (unsigned __int16 *)(a4 + 2), v17);
  if ( (__int64 *)v15[0] != &v16 )
    j_j___libc_free_0(v15[0]);
  if ( (v12 & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    *a1 = v12 & 0xFFFFFFFFFFFFFFFELL | 1;
  }
  else
  {
    v17[0].m128i_i64[0] = (__int64)"Type";
    v18 = 259;
    sub_37011E0((unsigned __int64 *)v15, v4, (unsigned int *)(a4 + 4), v17[0].m128i_i64);
    v6 = v15[0] & 0xFFFFFFFFFFFFFFFELL;
    if ( (v15[0] & 0xFFFFFFFFFFFFFFFELL) != 0
      || (v15[0] = 0,
          sub_9C66B0(v15),
          v17[0].m128i_i64[0] = (__int64)"FieldOffset",
          v18 = 259,
          sub_3702660((unsigned __int64 *)v15, v4, (unsigned __int64 *)(a4 + 8), v17, v7, v8),
          v6 = v15[0] & 0xFFFFFFFFFFFFFFFELL,
          (v15[0] & 0xFFFFFFFFFFFFFFFELL) != 0) )
    {
      v15[0] = 0;
      *a1 = v6 | 1;
      sub_9C66B0(v15);
    }
    else
    {
      v15[0] = 0;
      sub_9C66B0(v15);
      v17[0].m128i_i64[0] = (__int64)"Name";
      v18 = 259;
      sub_3701560((unsigned __int64 *)v15, v4, (_QWORD *)(a4 + 16), v17, v9);
      v10 = v15[0] & 0xFFFFFFFFFFFFFFFELL;
      if ( (v15[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
      {
        *a1 = 0;
        v15[0] = v10 | 1;
        sub_9C6670(a1, v15);
        sub_9C66B0(v15);
      }
      else
      {
        v15[0] = 0;
        sub_9C66B0(v15);
        *a1 = 1;
        v17[0].m128i_i64[0] = 0;
        sub_9C66B0(v17[0].m128i_i64);
      }
    }
  }
  if ( (__int64 *)v13[0] != &v14 )
    j_j___libc_free_0(v13[0]);
  return a1;
}
