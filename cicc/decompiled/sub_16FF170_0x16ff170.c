// Function: sub_16FF170
// Address: 0x16ff170
//
__int64 __fastcall sub_16FF170(_QWORD *a1, _QWORD *a2, const __m128i **a3)
{
  __int64 v5; // r12
  const __m128i *v6; // rax
  __int64 v7; // rax
  _QWORD *v8; // rdx
  __int64 v9; // r14
  _QWORD *v10; // r13
  _QWORD *v11; // rcx
  __int64 v12; // rdi
  unsigned __int64 v14; // r14
  unsigned __int64 v15; // r15
  const void *v16; // rsi
  int v17; // eax
  unsigned int v18; // edi
  int v19; // eax

  v5 = sub_22077B0(64);
  v6 = *a3;
  *(_QWORD *)(v5 + 48) = 0;
  *(_QWORD *)(v5 + 56) = 0;
  *(__m128i *)(v5 + 32) = _mm_loadu_si128(v6);
  v7 = sub_16FE940(a1, a2, v5 + 32);
  v9 = v7;
  if ( v8 )
  {
    v10 = v8;
    v11 = a1 + 1;
    v12 = 1;
    if ( v7 || v8 == v11 )
      goto LABEL_3;
    v14 = *(_QWORD *)(v5 + 40);
    v15 = v8[5];
    v16 = (const void *)v8[4];
    if ( v14 > v15 )
    {
      v12 = 0;
      if ( !v15 )
        goto LABEL_3;
      v19 = memcmp(*(const void **)(v5 + 32), v16, v8[5]);
      v11 = a1 + 1;
      v18 = v19;
      if ( !v19 )
        goto LABEL_9;
    }
    else if ( !v14
           || (v17 = memcmp(*(const void **)(v5 + 32), v16, *(_QWORD *)(v5 + 40)), v11 = a1 + 1, (v18 = v17) == 0) )
    {
      v12 = 0;
      if ( v14 == v15 )
      {
LABEL_3:
        sub_220F040(v12, v5, v10, v11);
        ++a1[5];
        return v5;
      }
LABEL_9:
      v12 = v14 < v15;
      goto LABEL_3;
    }
    v12 = v18 >> 31;
    goto LABEL_3;
  }
  j_j___libc_free_0(v5, 64);
  return v9;
}
