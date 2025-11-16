// Function: sub_370F770
// Address: 0x370f770
//
unsigned __int64 *__fastcall sub_370F770(unsigned __int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int64 v4; // rax
  unsigned __int64 v6; // [rsp+8h] [rbp-48h] BYREF
  __m128i v7[2]; // [rsp+10h] [rbp-40h] BYREF
  char v8; // [rsp+30h] [rbp-20h]
  char v9; // [rsp+31h] [rbp-1Fh]

  v7[0].m128i_i64[0] = (__int64)"Signature";
  v9 = 1;
  v8 = 3;
  sub_370BDF0(&v6, (_QWORD *)(a2 + 16), (unsigned int *)(a4 + 4), v7);
  v4 = v6 | 1;
  if ( (v6 & 0xFFFFFFFFFFFFFFFELL) == 0 )
    v4 = 1;
  *a1 = v4;
  return a1;
}
