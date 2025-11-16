// Function: sub_370D0A0
// Address: 0x370d0a0
//
unsigned __int64 *__fastcall sub_370D0A0(unsigned __int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned int v5; // r8d
  unsigned __int64 v7; // rax
  __int64 v8; // [rsp+8h] [rbp-68h] BYREF
  __m128i v9[2]; // [rsp+10h] [rbp-60h] BYREF
  char v10; // [rsp+30h] [rbp-40h]
  char v11; // [rsp+31h] [rbp-3Fh]

  v9[0].m128i_i64[0] = (__int64)"Id";
  v11 = 1;
  v10 = 3;
  sub_37011E0((unsigned __int64 *)&v8, (_QWORD *)(a2 + 16), (unsigned int *)(a4 + 2), v9[0].m128i_i64);
  if ( (v8 & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    *a1 = v8 & 0xFFFFFFFFFFFFFFFELL | 1;
  }
  else
  {
    v9[0].m128i_i64[0] = (__int64)"StringData";
    v11 = 1;
    v10 = 3;
    sub_3701560((unsigned __int64 *)&v8, (_QWORD *)(a2 + 16), (_QWORD *)(a4 + 8), v9, v5);
    if ( (v8 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    {
      v7 = v8 & 0xFFFFFFFFFFFFFFFELL | 1;
      v8 = 0;
      *a1 = v7;
      sub_9C66B0(&v8);
    }
    else
    {
      *a1 = 1;
    }
  }
  return a1;
}
