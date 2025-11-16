// Function: sub_E9C120
// Address: 0xe9c120
//
__int64 __fastcall sub_E9C120(_QWORD *a1, unsigned int a2, _QWORD *a3)
{
  __int64 result; // rax
  __int64 v7; // rbx
  __int64 v8; // rdx
  __int64 (*v9)(); // rax
  __m128i *v10; // rsi
  __int64 v11; // rdi
  const char *v12; // rax
  __m128i v13; // [rsp+0h] [rbp-50h] BYREF
  __int64 v14; // [rsp+10h] [rbp-40h]
  char v15; // [rsp+20h] [rbp-30h]
  char v16; // [rsp+21h] [rbp-2Fh]

  result = sub_E99590((__int64)a1, a3);
  if ( !result )
    return result;
  if ( !a2 )
  {
    v16 = 1;
    v11 = a1[1];
    v12 = "stack allocation size must be non-zero";
LABEL_13:
    v13.m128i_i64[0] = (__int64)v12;
    v15 = 3;
    return sub_E66880(v11, a3, (__int64)&v13);
  }
  if ( (a2 & 7) != 0 )
  {
    v16 = 1;
    v11 = a1[1];
    v12 = "stack allocation size is not a multiple of 8";
    goto LABEL_13;
  }
  v7 = result;
  v8 = 1;
  v9 = *(__int64 (**)())(*a1 + 88LL);
  if ( v9 != sub_E97650 )
    v8 = ((__int64 (__fastcall *)(_QWORD *, _QWORD *, __int64))v9)(a1, a3, 1);
  v13.m128i_i64[0] = v8;
  v13.m128i_i64[1] = a2 | 0xFFFFFFFF00000000LL;
  result = (unsigned int)(a2 < 0x81) + 1;
  LODWORD(v14) = (a2 < 0x81) + 1;
  v10 = *(__m128i **)(v7 + 96);
  if ( v10 == *(__m128i **)(v7 + 104) )
    return sub_E9B9B0((const __m128i **)(v7 + 88), v10, &v13);
  if ( v10 )
  {
    *v10 = _mm_loadu_si128(&v13);
    result = v14;
    v10[1].m128i_i64[0] = v14;
    v10 = *(__m128i **)(v7 + 96);
  }
  *(_QWORD *)(v7 + 96) = (char *)v10 + 24;
  return result;
}
