// Function: sub_F53E50
// Address: 0xf53e50
//
__int64 __fastcall sub_F53E50(unsigned __int8 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // r14
  int v10; // eax
  __int64 v11; // r15
  __int64 v12; // rbx
  int v13; // eax
  __int64 v15; // rdi
  unsigned int v16; // r14d
  unsigned int v17; // eax
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // rax
  __m128i v21; // xmm1
  __m128i v22; // xmm2
  __int64 v23; // rax
  __m128i v24; // [rsp+0h] [rbp-60h] BYREF
  __m128i v25; // [rsp+10h] [rbp-50h] BYREF
  __m128i v26[4]; // [rsp+20h] [rbp-40h] BYREF

  v9 = sub_B43CA0((__int64)a1) + 312;
  v10 = *a1;
  if ( (unsigned int)(v10 - 67) <= 0xC )
  {
    v11 = *((_QWORD *)a1 - 4);
    if ( sub_B507F0(a1, v9) )
      return v11;
    v12 = *((_QWORD *)a1 + 1);
    v13 = *(unsigned __int8 *)(v12 + 8);
    if ( (_BYTE)v13 == 14 )
    {
      v12 = sub_AE4450(v9, *((_QWORD *)a1 + 1));
      v13 = *(unsigned __int8 *)(v12 + 8);
    }
    if ( (unsigned int)(v13 - 17) > 1 && ((unsigned __int8)(*a1 - 76) <= 1u || (unsigned __int8)(*a1 - 67) <= 2u) )
    {
      v15 = *(_QWORD *)(v11 + 8);
      if ( *(_BYTE *)(v15 + 8) == 14 )
        v15 = sub_AE4450(v9, *(_QWORD *)(v11 + 8));
      v16 = sub_BCB060(v15);
      v17 = sub_BCB060(v12);
      sub_AF4FD0(&v24, v16, v17, *a1 == 69);
      v20 = *(unsigned int *)(a3 + 8);
      if ( v20 + 6 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
      {
        sub_C8D5F0(a3, (const void *)(a3 + 16), v20 + 6, 8u, v18, v19);
        v20 = *(unsigned int *)(a3 + 8);
      }
      v21 = _mm_loadu_si128(&v25);
      v22 = _mm_loadu_si128(v26);
      v23 = *(_QWORD *)a3 + 8 * v20;
      *(__m128i *)v23 = _mm_loadu_si128(&v24);
      *(__m128i *)(v23 + 16) = v21;
      *(__m128i *)(v23 + 32) = v22;
      *(_DWORD *)(a3 + 8) += 6;
      return v11;
    }
    return 0;
  }
  if ( (_BYTE)v10 == 63 )
    return sub_F53440((__int64)a1, v9, a2, a3, a4);
  if ( (unsigned int)(v10 - 42) <= 0x11 )
    return sub_F53AD0(a1, a2, a3, a4, v7, v8);
  if ( (_BYTE)v10 != 82 )
    return 0;
  return sub_F53C60((__int64)a1, a2, a3, a4, v7, v8);
}
