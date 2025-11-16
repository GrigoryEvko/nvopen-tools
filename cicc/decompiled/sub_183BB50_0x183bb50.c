// Function: sub_183BB50
// Address: 0x183bb50
//
unsigned __int64 __fastcall sub_183BB50(__int64 a1, int *a2, __int64 a3)
{
  int v6; // r14d
  __m128i *v7; // rdx
  unsigned __int64 result; // rax
  __m128i si128; // xmm0
  size_t v10; // rdx
  char *v11; // rsi
  const void *v12; // rdi
  const void *v13; // rsi
  size_t v14; // rdx
  __int64 v15; // rdx
  const void *v16; // rdi
  const void *v17; // rsi
  size_t v18; // rdx
  void *v19; // rdx
  const void *v20; // rdi
  const void *v21; // rsi
  size_t v22; // rdx
  __int64 v23; // rdx

  v6 = *a2;
  if ( *(_DWORD *)(a1 + 8) == *a2 )
  {
    v12 = (const void *)*((_QWORD *)a2 + 1);
    v13 = *(const void **)(a1 + 16);
    v14 = *((_QWORD *)a2 + 2) - (_QWORD)v12;
    if ( v14 == *(_QWORD *)(a1 + 24) - (_QWORD)v13 && (!v14 || !memcmp(v12, v13, v14)) )
    {
      v15 = *(_QWORD *)(a3 + 24);
      if ( (unsigned __int64)(*(_QWORD *)(a3 + 16) - v15) > 8 )
      {
        *(_BYTE *)(v15 + 8) = 100;
        *(_QWORD *)v15 = 0x656E696665646E75LL;
        *(_QWORD *)(a3 + 24) += 9LL;
        return 0x656E696665646E75LL;
      }
      v10 = 9;
      v11 = "undefined";
      return sub_16E7EE0(a3, v11, v10);
    }
  }
  if ( v6 == *(_DWORD *)(a1 + 40)
    && (v16 = (const void *)*((_QWORD *)a2 + 1),
        v17 = *(const void **)(a1 + 48),
        v18 = *((_QWORD *)a2 + 2) - (_QWORD)v16,
        v18 == *(_QWORD *)(a1 + 56) - (_QWORD)v17)
    && (!v18 || !memcmp(v16, v17, v18)) )
  {
    v19 = *(void **)(a3 + 24);
    if ( *(_QWORD *)(a3 + 16) - (_QWORD)v19 <= 0xAu )
    {
      v10 = 11;
      v11 = "overdefined";
      return sub_16E7EE0(a3, v11, v10);
    }
    qmemcpy(v19, "overdefined", 11);
    *(_QWORD *)(a3 + 24) += 11LL;
    return 25966;
  }
  else
  {
    if ( v6 != *(_DWORD *)(a1 + 72)
      || (v20 = (const void *)*((_QWORD *)a2 + 1),
          v21 = *(const void **)(a1 + 80),
          v22 = *((_QWORD *)a2 + 2) - (_QWORD)v20,
          v22 != *(_QWORD *)(a1 + 88) - (_QWORD)v21)
      || v22 && memcmp(v20, v21, v22) )
    {
      v7 = *(__m128i **)(a3 + 24);
      result = *(_QWORD *)(a3 + 16) - (_QWORD)v7;
      if ( result > 0x14 )
      {
        si128 = _mm_load_si128((const __m128i *)&xmmword_42B9810);
        v7[1].m128i_i32[0] = 1970037110;
        v7[1].m128i_i8[4] = 101;
        *v7 = si128;
        *(_QWORD *)(a3 + 24) += 21LL;
        return result;
      }
      v10 = 21;
      v11 = "unknown lattice value";
      return sub_16E7EE0(a3, v11, v10);
    }
    v23 = *(_QWORD *)(a3 + 24);
    if ( (unsigned __int64)(*(_QWORD *)(a3 + 16) - v23) <= 8 )
    {
      v10 = 9;
      v11 = "untracked";
      return sub_16E7EE0(a3, v11, v10);
    }
    *(_BYTE *)(v23 + 8) = 100;
    *(_QWORD *)v23 = 0x656B636172746E75LL;
    *(_QWORD *)(a3 + 24) += 9LL;
    return 0x656B636172746E75LL;
  }
}
