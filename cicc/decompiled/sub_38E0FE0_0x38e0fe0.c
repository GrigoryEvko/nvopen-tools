// Function: sub_38E0FE0
// Address: 0x38e0fe0
//
unsigned __int64 __fastcall sub_38E0FE0(_QWORD *a1, unsigned int a2, unsigned __int64 a3)
{
  unsigned __int64 result; // rax
  unsigned __int64 *v7; // rbx
  __int64 v8; // rdx
  __int64 (*v9)(); // rax
  __m128i *v10; // rsi
  __int64 v11; // rdi
  const char *v12; // rax
  __m128i v13; // [rsp+0h] [rbp-40h] BYREF
  unsigned __int64 v14; // [rsp+10h] [rbp-30h]

  result = sub_38DD280((__int64)a1, a3);
  if ( !result )
    return result;
  if ( !a2 )
  {
    BYTE1(v14) = 1;
    v11 = a1[1];
    v12 = "stack allocation size must be non-zero";
LABEL_13:
    v13.m128i_i64[0] = (__int64)v12;
    LOBYTE(v14) = 3;
    return (unsigned __int64)sub_38BE3D0(v11, a3, (__int64)&v13);
  }
  if ( (a2 & 7) != 0 )
  {
    BYTE1(v14) = 1;
    v11 = a1[1];
    v12 = "stack allocation size is not a multiple of 8";
    goto LABEL_13;
  }
  v7 = (unsigned __int64 *)result;
  v8 = 1;
  v9 = *(__int64 (**)())(*a1 + 16LL);
  if ( v9 != sub_38DBC10 )
    v8 = ((__int64 (__fastcall *)(_QWORD *, unsigned __int64, __int64))v9)(a1, a3, 1);
  v13.m128i_i64[0] = v8;
  v13.m128i_i64[1] = a2 | 0xFFFFFFFF00000000LL;
  result = (unsigned int)(a2 < 0x81) + 1;
  LODWORD(v14) = (a2 < 0x81) + 1;
  v10 = (__m128i *)v7[10];
  if ( v10 == (__m128i *)v7[11] )
    return sub_38E08D0(v7 + 9, v10, &v13);
  if ( v10 )
  {
    *v10 = _mm_loadu_si128(&v13);
    result = v14;
    v10[1].m128i_i64[0] = v14;
    v10 = (__m128i *)v7[10];
  }
  v7[10] = (unsigned __int64)&v10[1].m128i_u64[1];
  return result;
}
