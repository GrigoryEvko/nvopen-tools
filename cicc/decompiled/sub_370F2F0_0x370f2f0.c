// Function: sub_370F2F0
// Address: 0x370f2f0
//
__int64 *__fastcall sub_370F2F0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  _QWORD *v4; // r12
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rax
  unsigned __int64 v9; // [rsp+8h] [rbp-68h] BYREF
  __m128i v10[2]; // [rsp+10h] [rbp-60h] BYREF
  char v11; // [rsp+30h] [rbp-40h]
  char v12; // [rsp+31h] [rbp-3Fh]

  v4 = (_QWORD *)(a2 + 16);
  v10[0].m128i_i64[0] = (__int64)"UDT";
  v12 = 1;
  v11 = 3;
  sub_37011E0(&v9, (_QWORD *)(a2 + 16), (unsigned int *)(a4 + 2), v10[0].m128i_i64);
  if ( (v9 & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    *a1 = v9 & 0xFFFFFFFFFFFFFFFELL | 1;
  }
  else
  {
    v10[0].m128i_i64[0] = (__int64)"SourceFile";
    v12 = 1;
    v11 = 3;
    sub_37011E0(&v9, v4, (unsigned int *)(a4 + 6), v10[0].m128i_i64);
    v6 = v9 & 0xFFFFFFFFFFFFFFFELL;
    if ( (v9 & 0xFFFFFFFFFFFFFFFELL) != 0
      || (v9 = 0,
          sub_9C66B0((__int64 *)&v9),
          v12 = 1,
          v10[0].m128i_i64[0] = (__int64)"LineNumber",
          v11 = 3,
          sub_370BDF0(&v9, v4, (unsigned int *)(a4 + 12), v10),
          v6 = v9 & 0xFFFFFFFFFFFFFFFELL,
          (v9 & 0xFFFFFFFFFFFFFFFELL) != 0) )
    {
      v9 = 0;
      *a1 = v6 | 1;
      sub_9C66B0((__int64 *)&v9);
    }
    else
    {
      v9 = 0;
      sub_9C66B0((__int64 *)&v9);
      v12 = 1;
      v10[0].m128i_i64[0] = (__int64)"Module";
      v11 = 3;
      sub_370BC10(&v9, v4, (unsigned __int16 *)(a4 + 16), v10);
      v7 = v9 & 0xFFFFFFFFFFFFFFFELL;
      if ( (v9 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      {
        *a1 = 0;
        v9 = v7 | 1;
        sub_9C6670(a1, &v9);
        sub_9C66B0((__int64 *)&v9);
      }
      else
      {
        v9 = 0;
        sub_9C66B0((__int64 *)&v9);
        *a1 = 1;
        v10[0].m128i_i64[0] = 0;
        sub_9C66B0(v10[0].m128i_i64);
      }
    }
  }
  return a1;
}
