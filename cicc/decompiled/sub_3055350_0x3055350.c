// Function: sub_3055350
// Address: 0x3055350
//
__int64 __fastcall sub_3055350(__int64 a1, __int64 a2, __m128i *a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v7; // rcx
  __int64 v8; // r8
  __m128i *v9; // r12
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v13; // rax
  char v14; // dl
  const __m128i *v15; // r14
  __int64 v16; // rax
  __int64 v17; // rdx
  bool v18; // r10
  __m128i *v19; // rax
  _QWORD *v20; // rcx
  __int64 v21; // rax
  __m128i si128; // xmm0
  __int64 v23; // rdx
  __int64 v24; // rax
  unsigned __int64 v25; // rax
  _QWORD *v26; // [rsp+8h] [rbp-58h]
  char v27; // [rsp+14h] [rbp-4Ch]
  _QWORD *v28; // [rsp+18h] [rbp-48h]
  __m128i v29[4]; // [rsp+20h] [rbp-40h] BYREF

  if ( *(_QWORD *)(a2 + 120) )
  {
    v13 = sub_3055130(a2 + 80, a3);
    *(_BYTE *)(a1 + 8) = 0;
    *(_QWORD *)a1 = v13;
    *(_BYTE *)(a1 + 16) = v14;
  }
  else
  {
    v7 = *(unsigned int *)(a2 + 8);
    v8 = *(_QWORD *)a2;
    v9 = (__m128i *)(*(_QWORD *)a2 + 16 * v7);
    if ( *(__m128i **)a2 == v9 )
    {
      if ( v7 > 3 )
      {
        v28 = (_QWORD *)(a2 + 80);
LABEL_19:
        *(_DWORD *)(a2 + 8) = 0;
        v21 = sub_3055130((__int64)v28, a3);
        *(_BYTE *)(a1 + 8) = 0;
        *(_QWORD *)a1 = v21;
        *(_BYTE *)(a1 + 16) = 1;
        return a1;
      }
    }
    else
    {
      v10 = a3->m128i_i64[0];
      v11 = *(_QWORD *)a2;
      while ( *(_QWORD *)v11 != v10 || *(_DWORD *)(v11 + 8) != a3->m128i_i32[2] )
      {
        v11 += 16;
        if ( v9 == (__m128i *)v11 )
          goto LABEL_11;
      }
      if ( v9 != (__m128i *)v11 )
      {
        *(_BYTE *)(a1 + 8) = 1;
        *(_QWORD *)a1 = v11;
        *(_BYTE *)(a1 + 16) = 0;
        return a1;
      }
LABEL_11:
      if ( v7 > 3 )
      {
        v15 = *(const __m128i **)a2;
        v28 = (_QWORD *)(a2 + 80);
        v29[0].m128i_i64[0] = a2 + 88;
        do
        {
          v16 = sub_3055200(v28, v29[0].m128i_i64[0], (__int64)v15);
          if ( v17 )
          {
            v18 = 1;
            if ( !v16 && v17 != v29[0].m128i_i64[0] )
            {
              v25 = *(_QWORD *)(v17 + 32);
              if ( v15->m128i_i64[0] >= v25 && (v15->m128i_i64[0] != v25 || v15->m128i_i32[2] >= *(_DWORD *)(v17 + 40)) )
                v18 = 0;
            }
            v26 = (_QWORD *)v17;
            v27 = v18;
            v19 = (__m128i *)sub_22077B0(0x30u);
            v20 = (_QWORD *)v29[0].m128i_i64[0];
            v19[2] = _mm_loadu_si128(v15);
            sub_220F040(v27, (__int64)v19, v26, v20);
            ++*(_QWORD *)(a2 + 120);
          }
          ++v15;
        }
        while ( v9 != v15 );
        goto LABEL_19;
      }
    }
    si128 = _mm_loadu_si128(a3);
    if ( v7 + 1 > *(unsigned int *)(a2 + 12) )
    {
      v29[0] = si128;
      sub_C8D5F0(a2, (const void *)(a2 + 16), v7 + 1, 0x10u, v8, a6);
      si128 = _mm_load_si128(v29);
      v9 = (__m128i *)(*(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8));
    }
    *v9 = si128;
    v23 = *(_QWORD *)a2;
    v24 = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
    *(_DWORD *)(a2 + 8) = v24;
    *(_BYTE *)(a1 + 8) = 1;
    *(_QWORD *)a1 = v23 + 16 * v24 - 16;
    *(_BYTE *)(a1 + 16) = 1;
  }
  return a1;
}
