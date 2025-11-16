// Function: sub_1EFA190
// Address: 0x1efa190
//
void __fastcall sub_1EFA190(__int64 *a1, __m128i *a2, __int64 a3)
{
  __int64 v3; // r14
  __int64 v6; // r12
  __int64 v7; // rax
  const __m128i *v8; // rdx
  __m128i v9; // xmm1
  __int64 v10; // rdi
  __int64 v11; // r8
  __int32 v12; // esi
  __int32 v13; // ecx
  __int32 v14; // eax
  char *v15; // rax
  __m128i v16; // xmm0
  int v17; // ecx
  unsigned __int64 v18; // r12
  __int8 *v19; // rcx
  __int32 v20; // ecx
  __m128i v21; // xmm2
  const __m128i *v22; // rax

  v3 = 0x333333333333333LL;
  if ( a3 <= 0x333333333333333LL )
    v3 = a3;
  *a1 = a3;
  a1[1] = 0;
  a1[2] = 0;
  if ( a3 > 0 )
  {
    while ( 1 )
    {
      v6 = 40 * v3;
      v7 = sub_2207800(40 * v3, &unk_435FF63);
      v8 = (const __m128i *)v7;
      if ( v7 )
        break;
      v3 >>= 1;
      if ( !v3 )
        return;
    }
    v9 = _mm_loadu_si128(a2 + 1);
    v10 = a2->m128i_i64[0];
    v11 = v7 + v6;
    a2[1].m128i_i64[0] = 0;
    v12 = a2->m128i_i32[2];
    v13 = a2->m128i_i32[3];
    a2[1].m128i_i64[1] = 0;
    *(_QWORD *)v7 = v10;
    *(_DWORD *)(v7 + 8) = v12;
    *(_DWORD *)(v7 + 12) = v13;
    *(__m128i *)(v7 + 16) = v9;
    v14 = a2[2].m128i_i32[0];
    a2[2].m128i_i32[0] = 0;
    v8[2].m128i_i32[0] = v14;
    v15 = &v8[2].m128i_i8[8];
    if ( (unsigned __int64 *)v11 == &v8[2].m128i_u64[1] )
    {
      v22 = v8;
    }
    else
    {
      while ( 1 )
      {
        v16 = _mm_loadu_si128((const __m128i *)(v15 - 24));
        *((_DWORD *)v15 + 3) = v13;
        v15 += 40;
        v17 = *((_DWORD *)v15 - 12);
        *((_QWORD *)v15 - 5) = v10;
        *((_DWORD *)v15 - 8) = v12;
        *(__m128i *)(v15 - 24) = v16;
        *((_DWORD *)v15 - 2) = v17;
        *((_QWORD *)v15 - 8) = 0;
        *((_QWORD *)v15 - 7) = 0;
        *((_DWORD *)v15 - 12) = 0;
        if ( (char *)v11 == v15 )
          break;
        v10 = *((_QWORD *)v15 - 5);
        v12 = *((_DWORD *)v15 - 8);
        v13 = *((_DWORD *)v15 - 7);
      }
      v18 = (0xCCCCCCCCCCCCCCDLL * ((unsigned __int64)(v6 - 80) >> 3)) & 0x1FFFFFFFFFFFFFFFLL;
      v19 = &v8->m128i_i8[40 * v18];
      v22 = (const __m128i *)((char *)v8 + 40 * v18 + 40);
      v10 = *((_QWORD *)v19 + 5);
      v12 = *((_DWORD *)v19 + 12);
      v13 = *((_DWORD *)v19 + 13);
    }
    a2->m128i_i32[2] = v12;
    a2->m128i_i32[3] = v13;
    a2->m128i_i64[0] = v10;
    if ( &v22[1] != &a2[1] )
    {
      v20 = v22[2].m128i_i32[0];
      v21 = _mm_loadu_si128(v22 + 1);
      v22[2].m128i_i32[0] = 0;
      v22[1].m128i_i64[0] = 0;
      a2[2].m128i_i32[0] = v20;
      v22[1].m128i_i64[1] = 0;
      a2[1] = v21;
    }
    a1[2] = (__int64)v8;
    a1[1] = v3;
  }
}
