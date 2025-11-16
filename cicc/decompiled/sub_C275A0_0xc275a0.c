// Function: sub_C275A0
// Address: 0xc275a0
//
__int64 __fastcall sub_C275A0(_QWORD *a1, _QWORD *a2, const __m128i **a3)
{
  __int64 v5; // r12
  _QWORD *v6; // rax
  _QWORD *v7; // rdx
  _QWORD *v8; // r14
  _QWORD *v9; // r13
  _QWORD *v10; // rcx
  _BOOL8 v11; // rdi
  unsigned __int64 v13; // r15
  unsigned __int64 v14; // r14
  const void *v15; // rsi
  const void *v16; // rdi
  size_t v17; // rdx
  unsigned int v18; // eax

  v5 = sub_22077B0(224);
  *(__m128i *)(v5 + 32) = _mm_loadu_si128(*a3);
  memset((void *)(v5 + 48), 0, 0xB0u);
  *(_QWORD *)(v5 + 144) = v5 + 128;
  *(_QWORD *)(v5 + 152) = v5 + 128;
  *(_QWORD *)(v5 + 192) = v5 + 176;
  *(_QWORD *)(v5 + 200) = v5 + 176;
  v6 = sub_C1C960(a1, a2, (const void **)(v5 + 32));
  v8 = v6;
  if ( v7 )
  {
    v9 = v7;
    v10 = a1 + 1;
    v11 = 1;
    if ( v6 || v7 == v10 )
      goto LABEL_3;
    v13 = v7[5];
    v14 = *(_QWORD *)(v5 + 40);
    v15 = (const void *)v7[4];
    v16 = *(const void **)(v5 + 32);
    if ( v13 < v14 )
    {
      if ( v16 == v15 )
        goto LABEL_12;
      v17 = v7[5];
    }
    else
    {
      if ( v16 == v15 )
      {
LABEL_11:
        v11 = 0;
        if ( v13 != v14 )
LABEL_12:
          v11 = v13 > v14;
LABEL_3:
        sub_220F040(v11, v5, v9, v10);
        ++a1[5];
        return v5;
      }
      v17 = *(_QWORD *)(v5 + 40);
    }
    if ( !v16 )
    {
      v11 = 1;
      goto LABEL_3;
    }
    if ( !v15 )
    {
      v11 = 0;
      goto LABEL_3;
    }
    v18 = memcmp(v16, v15, v17);
    v10 = a1 + 1;
    v11 = v18 >> 31;
    if ( v18 )
      goto LABEL_3;
    goto LABEL_11;
  }
  sub_C1EF60(0);
  j_j___libc_free_0(v5, 224);
  return (__int64)v8;
}
