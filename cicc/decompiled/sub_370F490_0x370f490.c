// Function: sub_370F490
// Address: 0x370f490
//
unsigned __int64 *__fastcall sub_370F490(unsigned __int64 *a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5)
{
  _QWORD *v5; // r12
  unsigned __int64 v7; // rax
  unsigned int v8; // r8d
  __int64 v10; // [rsp+8h] [rbp-68h] BYREF
  __m128i v11[2]; // [rsp+10h] [rbp-60h] BYREF
  char v12; // [rsp+30h] [rbp-40h]
  char v13; // [rsp+31h] [rbp-3Fh]

  v5 = (_QWORD *)(a2 + 16);
  v11[0].m128i_i64[0] = (__int64)"Guid";
  v13 = 1;
  v12 = 3;
  sub_37016C0((unsigned __int64 *)&v10, (_QWORD *)(a2 + 16), (__m128i *)(a4 + 2), v11, a5);
  if ( (v10 & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    *a1 = v10 & 0xFFFFFFFFFFFFFFFELL | 1;
  }
  else
  {
    v11[0].m128i_i64[0] = (__int64)"Age";
    v13 = 1;
    v12 = 3;
    sub_370BDF0((unsigned __int64 *)&v10, v5, (unsigned int *)(a4 + 20), v11);
    v7 = v10 & 0xFFFFFFFFFFFFFFFELL;
    if ( (v10 & 0xFFFFFFFFFFFFFFFELL) != 0
      || (v10 = 0,
          sub_9C66B0(&v10),
          v13 = 1,
          v11[0].m128i_i64[0] = (__int64)"Name",
          v12 = 3,
          sub_3701560((unsigned __int64 *)&v10, v5, (_QWORD *)(a4 + 24), v11, v8),
          v7 = v10 & 0xFFFFFFFFFFFFFFFELL,
          (v10 & 0xFFFFFFFFFFFFFFFELL) != 0) )
    {
      v10 = 0;
      *a1 = v7 | 1;
      sub_9C66B0(&v10);
    }
    else
    {
      v10 = 0;
      sub_9C66B0(&v10);
      *a1 = 1;
      v11[0].m128i_i64[0] = 0;
      sub_9C66B0(v11[0].m128i_i64);
    }
  }
  return a1;
}
