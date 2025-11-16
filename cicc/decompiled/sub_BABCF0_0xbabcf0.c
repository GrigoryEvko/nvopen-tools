// Function: sub_BABCF0
// Address: 0xbabcf0
//
void __fastcall sub_BABCF0(__int64 a1, const __m128i *a2)
{
  __m128i *v2; // rdx
  __int64 v3; // r14
  const void *v4; // r15
  signed __int64 v5; // r12
  __int64 v6; // rcx
  __int64 v7; // rax
  bool v8; // cf
  unsigned __int64 v9; // rax
  char *v10; // r14
  char *v11; // rcx
  __m128i *v12; // rax
  __m128i v13; // xmm3
  __int64 v14; // r13
  __int64 v15; // rsi
  __int64 v16; // r14
  char *v17; // [rsp+8h] [rbp-38h]

  v2 = *(__m128i **)(a1 + 8);
  if ( v2 == *(__m128i **)(a1 + 16) )
  {
    v3 = 0x3FFFFFFFFFFFFFFLL;
    v4 = *(const void **)a1;
    v5 = (signed __int64)v2->m128i_i64 - *(_QWORD *)a1;
    v6 = v5 >> 5;
    if ( v5 >> 5 == 0x3FFFFFFFFFFFFFFLL )
      sub_4262D8((__int64)"vector::_M_realloc_insert");
    v7 = 1;
    if ( v6 )
      v7 = v5 >> 5;
    v8 = __CFADD__(v6, v7);
    v9 = v6 + v7;
    if ( v8 )
    {
      v16 = 0x7FFFFFFFFFFFFFE0LL;
    }
    else
    {
      if ( !v9 )
      {
        v10 = 0;
        v11 = 0;
        goto LABEL_11;
      }
      if ( v9 <= 0x3FFFFFFFFFFFFFFLL )
        v3 = v9;
      v16 = 32 * v3;
    }
    v11 = (char *)sub_22077B0(v16);
    v10 = &v11[v16];
LABEL_11:
    v12 = (__m128i *)&v11[v5];
    if ( &v11[v5] )
    {
      v13 = _mm_loadu_si128(a2 + 1);
      *v12 = _mm_loadu_si128(a2);
      v12[1] = v13;
    }
    v14 = (__int64)&v11[v5 + 32];
    if ( v5 > 0 )
    {
      v11 = (char *)memmove(v11, v4, v5);
      v15 = *(_QWORD *)(a1 + 16) - (_QWORD)v4;
    }
    else
    {
      if ( !v4 )
      {
LABEL_15:
        *(_QWORD *)(a1 + 8) = v14;
        *(_QWORD *)(a1 + 16) = v10;
        *(_QWORD *)a1 = v11;
        return;
      }
      v15 = *(_QWORD *)(a1 + 16) - (_QWORD)v4;
    }
    v17 = v11;
    j_j___libc_free_0(v4, v15);
    v11 = v17;
    goto LABEL_15;
  }
  if ( v2 )
  {
    *v2 = _mm_loadu_si128(a2);
    v2[1] = _mm_loadu_si128(a2 + 1);
    v2 = *(__m128i **)(a1 + 8);
  }
  *(_QWORD *)(a1 + 8) = v2 + 2;
}
