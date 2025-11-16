// Function: sub_3714340
// Address: 0x3714340
//
__int64 *__fastcall sub_3714340(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  _QWORD *v4; // r12
  unsigned __int64 v6; // rax
  __int64 v7; // r8
  __int64 v8; // r9
  unsigned __int64 v9; // rax
  __int64 v10; // r8
  __int64 v11; // r9
  unsigned __int64 v13; // [rsp+8h] [rbp-A8h] BYREF
  __int64 v14[2]; // [rsp+10h] [rbp-A0h] BYREF
  __int64 v15; // [rsp+20h] [rbp-90h] BYREF
  __int64 v16[2]; // [rsp+30h] [rbp-80h] BYREF
  __int64 v17; // [rsp+40h] [rbp-70h] BYREF
  __m128i v18[2]; // [rsp+50h] [rbp-60h] BYREF
  __int16 v19; // [rsp+70h] [rbp-40h]

  v4 = (_QWORD *)(a2 + 16);
  sub_37128E0(v14, (_QWORD *)(a2 + 16), *(_BYTE *)(a4 + 2) & 3, 0, 0);
  sub_8FD6D0((__int64)v16, "Attrs: ", v14);
  v19 = 260;
  v18[0].m128i_i64[0] = (__int64)v16;
  sub_370BC10(&v13, (_QWORD *)(a2 + 16), (unsigned __int16 *)(a4 + 2), v18);
  if ( (__int64 *)v16[0] != &v17 )
    j_j___libc_free_0(v16[0]);
  if ( (v13 & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    *a1 = v13 & 0xFFFFFFFFFFFFFFFELL | 1;
  }
  else
  {
    v18[0].m128i_i64[0] = (__int64)"BaseType";
    v19 = 259;
    sub_37011E0((unsigned __int64 *)v16, v4, (unsigned int *)(a4 + 4), v18[0].m128i_i64);
    v6 = v16[0] & 0xFFFFFFFFFFFFFFFELL;
    if ( (v16[0] & 0xFFFFFFFFFFFFFFFELL) != 0
      || (v16[0] = 0,
          sub_9C66B0(v16),
          v18[0].m128i_i64[0] = (__int64)"VBPtrType",
          v19 = 259,
          sub_37011E0((unsigned __int64 *)v16, v4, (unsigned int *)(a4 + 8), v18[0].m128i_i64),
          v6 = v16[0] & 0xFFFFFFFFFFFFFFFELL,
          (v16[0] & 0xFFFFFFFFFFFFFFFELL) != 0) )
    {
      v16[0] = 0;
      *a1 = v6 | 1;
      sub_9C66B0(v16);
    }
    else
    {
      v16[0] = 0;
      sub_9C66B0(v16);
      v18[0].m128i_i64[0] = (__int64)"VBPtrOffset";
      v19 = 259;
      sub_3702660((unsigned __int64 *)v16, v4, (unsigned __int64 *)(a4 + 16), v18, v7, v8);
      v9 = v16[0] & 0xFFFFFFFFFFFFFFFELL;
      if ( (v16[0] & 0xFFFFFFFFFFFFFFFELL) != 0
        || (v16[0] = 0,
            sub_9C66B0(v16),
            v18[0].m128i_i64[0] = (__int64)"VBTableIndex",
            v19 = 259,
            sub_3702660((unsigned __int64 *)v16, v4, (unsigned __int64 *)(a4 + 24), v18, v10, v11),
            v9 = v16[0] & 0xFFFFFFFFFFFFFFFELL,
            (v16[0] & 0xFFFFFFFFFFFFFFFELL) != 0) )
      {
        *a1 = 0;
        v16[0] = v9 | 1;
        sub_9C6670(a1, v16);
        sub_9C66B0(v16);
      }
      else
      {
        v16[0] = 0;
        sub_9C66B0(v16);
        *a1 = 1;
        v18[0].m128i_i64[0] = 0;
        sub_9C66B0(v18[0].m128i_i64);
      }
    }
  }
  if ( (__int64 *)v14[0] != &v15 )
    j_j___libc_free_0(v14[0]);
  return a1;
}
