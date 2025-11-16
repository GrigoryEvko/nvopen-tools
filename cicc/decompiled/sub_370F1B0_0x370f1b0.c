// Function: sub_370F1B0
// Address: 0x370f1b0
//
unsigned __int64 *__fastcall sub_370F1B0(unsigned __int64 *a1, __int64 a2, __int64 a3, unsigned int *a4)
{
  _QWORD *v4; // r12
  unsigned __int64 v6; // rax
  __int64 v8; // [rsp+8h] [rbp-68h] BYREF
  __m128i v9[2]; // [rsp+10h] [rbp-60h] BYREF
  char v10; // [rsp+30h] [rbp-40h]
  char v11; // [rsp+31h] [rbp-3Fh]

  v4 = (_QWORD *)(a2 + 16);
  v9[0].m128i_i64[0] = (__int64)"UDT";
  v11 = 1;
  v10 = 3;
  sub_37011E0((unsigned __int64 *)&v8, (_QWORD *)(a2 + 16), (unsigned int *)((char *)a4 + 2), v9[0].m128i_i64);
  if ( (v8 & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    *a1 = v8 & 0xFFFFFFFFFFFFFFFELL | 1;
  }
  else
  {
    v9[0].m128i_i64[0] = (__int64)"SourceFile";
    v11 = 1;
    v10 = 3;
    sub_37011E0((unsigned __int64 *)&v8, v4, (unsigned int *)((char *)a4 + 6), v9[0].m128i_i64);
    v6 = v8 & 0xFFFFFFFFFFFFFFFELL;
    if ( (v8 & 0xFFFFFFFFFFFFFFFELL) != 0
      || (v8 = 0,
          sub_9C66B0(&v8),
          v11 = 1,
          v9[0].m128i_i64[0] = (__int64)"LineNumber",
          v10 = 3,
          sub_370BDF0((unsigned __int64 *)&v8, v4, a4 + 3, v9),
          v6 = v8 & 0xFFFFFFFFFFFFFFFELL,
          (v8 & 0xFFFFFFFFFFFFFFFELL) != 0) )
    {
      v8 = 0;
      *a1 = v6 | 1;
      sub_9C66B0(&v8);
    }
    else
    {
      v8 = 0;
      sub_9C66B0(&v8);
      *a1 = 1;
      v9[0].m128i_i64[0] = 0;
      sub_9C66B0(v9[0].m128i_i64);
    }
  }
  return a1;
}
