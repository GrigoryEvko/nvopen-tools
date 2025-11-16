// Function: sub_370D330
// Address: 0x370d330
//
__int64 *__fastcall sub_370D330(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  _QWORD *v4; // r12
  unsigned __int64 v6; // rax
  __int64 v7; // r8
  __int64 v8; // r9
  unsigned int v9; // r8d
  unsigned __int64 v10; // rax
  unsigned __int64 v12; // [rsp+8h] [rbp-68h] BYREF
  __m128i v13[2]; // [rsp+10h] [rbp-60h] BYREF
  char v14; // [rsp+30h] [rbp-40h]
  char v15; // [rsp+31h] [rbp-3Fh]

  v4 = (_QWORD *)(a2 + 16);
  v13[0].m128i_i64[0] = (__int64)"ElementType";
  v15 = 1;
  v14 = 3;
  sub_37011E0(&v12, (_QWORD *)(a2 + 16), (unsigned int *)(a4 + 2), v13[0].m128i_i64);
  if ( (v12 & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    *a1 = v12 & 0xFFFFFFFFFFFFFFFELL | 1;
  }
  else
  {
    v13[0].m128i_i64[0] = (__int64)"IndexType";
    v15 = 1;
    v14 = 3;
    sub_37011E0(&v12, v4, (unsigned int *)(a4 + 6), v13[0].m128i_i64);
    v6 = v12 & 0xFFFFFFFFFFFFFFFELL;
    if ( (v12 & 0xFFFFFFFFFFFFFFFELL) != 0
      || (v12 = 0,
          sub_9C66B0((__int64 *)&v12),
          v15 = 1,
          v13[0].m128i_i64[0] = (__int64)"SizeOf",
          v14 = 3,
          sub_3702660(&v12, v4, (unsigned __int64 *)(a4 + 16), v13, v7, v8),
          v6 = v12 & 0xFFFFFFFFFFFFFFFELL,
          (v12 & 0xFFFFFFFFFFFFFFFELL) != 0) )
    {
      v12 = 0;
      *a1 = v6 | 1;
      sub_9C66B0((__int64 *)&v12);
    }
    else
    {
      v12 = 0;
      sub_9C66B0((__int64 *)&v12);
      v15 = 1;
      v13[0].m128i_i64[0] = (__int64)"Name";
      v14 = 3;
      sub_3701560(&v12, v4, (_QWORD *)(a4 + 24), v13, v9);
      v10 = v12 & 0xFFFFFFFFFFFFFFFELL;
      if ( (v12 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      {
        *a1 = 0;
        v12 = v10 | 1;
        sub_9C6670(a1, &v12);
        sub_9C66B0((__int64 *)&v12);
      }
      else
      {
        v12 = 0;
        sub_9C66B0((__int64 *)&v12);
        *a1 = 1;
        v13[0].m128i_i64[0] = 0;
        sub_9C66B0(v13[0].m128i_i64);
      }
    }
  }
  return a1;
}
