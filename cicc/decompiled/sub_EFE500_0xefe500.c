// Function: sub_EFE500
// Address: 0xefe500
//
__m128i **__fastcall sub_EFE500(
        __m128i **a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        __int64 a8)
{
  int v10; // r15d
  __int64 v11; // rax
  __m128i *v12; // rbx
  __m128i v13; // xmm1
  __int64 v14; // r13
  __int64 v15; // rdi
  __int64 v16; // rax
  __m128i v18; // [rsp+20h] [rbp-50h] BYREF
  __int64 v19; // [rsp+30h] [rbp-40h]

  v10 = *(_DWORD *)(a2 + 1904);
  v18 = _mm_loadu_si128((const __m128i *)&a7);
  if ( v10 != 2 )
    v10 = 0;
  v19 = a8;
  v11 = sub_22077B0(1880);
  v12 = (__m128i *)v11;
  if ( v11 )
  {
    *(_QWORD *)(v11 + 8) = a3;
    v13 = _mm_loadu_si128(&v18);
    v14 = v11 + 16;
    v15 = v11 + 16;
    *(_BYTE *)(v11 + 1824) = 0;
    *(_QWORD *)(v11 + 1832) = 0;
    *(_QWORD *)(v11 + 1840) = a2 + 32;
    *(_QWORD *)v11 = &unk_49E4EE8;
    v16 = v19;
    v12[115].m128i_i8[8] = 1;
    v12[117].m128i_i64[0] = v16;
    v12[116] = v13;
    sub_EFDDC0(v15, v10);
    v12[114].m128i_i8[0] = 1;
    v12[114].m128i_i64[1] = v14;
  }
  *a1 = v12;
  return a1;
}
