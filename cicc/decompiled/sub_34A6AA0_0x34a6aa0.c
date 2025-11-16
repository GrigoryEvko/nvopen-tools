// Function: sub_34A6AA0
// Address: 0x34a6aa0
//
__int64 __fastcall sub_34A6AA0(__int64 a1, __int64 a2, __m128i *a3, __int64 a4, __int64 a5)
{
  __int64 v6; // r13
  _QWORD *v7; // rdx
  __m128i *v8; // rsi
  char v9; // al
  unsigned __int64 v11; // rcx
  __int64 v12; // r9
  __m128i *v13; // r13
  __int64 v14; // rdx
  __m128i *v15; // rax
  __m128i si128; // xmm0
  __int64 v17; // rdx
  __int64 v18; // rax
  __m128i *v19; // r14
  _QWORD *v20; // rax
  _QWORD *v21; // rdx
  bool v22; // r10
  __m128i *v23; // rax
  _QWORD *v24; // rcx
  __m128i *v25; // rax
  _QWORD *v26; // rdx
  unsigned __int64 v27; // rax
  _QWORD *v28; // [rsp+8h] [rbp-58h]
  char v29; // [rsp+14h] [rbp-4Ch]
  _QWORD *v30; // [rsp+18h] [rbp-48h]
  __m128i v31[4]; // [rsp+20h] [rbp-40h] BYREF

  if ( *(_QWORD *)(a2 + 120) )
  {
    v6 = a2 + 80;
    v8 = (__m128i *)sub_34A55B0(a2 + 80, (unsigned __int64 *)a3);
    v9 = 0;
    if ( v7 )
    {
      v8 = sub_349EB00(v6, (__int64)v8, v7, a3);
      v9 = 1;
    }
    *(_BYTE *)(a1 + 8) = 0;
    *(_QWORD *)a1 = v8;
    *(_BYTE *)(a1 + 16) = v9;
    return a1;
  }
  v11 = *(unsigned int *)(a2 + 8);
  v12 = *(_QWORD *)a2;
  v13 = (__m128i *)(*(_QWORD *)a2 + 16 * v11);
  if ( *(__m128i **)a2 == v13 )
  {
    if ( v11 <= 3 )
    {
LABEL_13:
      si128 = _mm_loadu_si128(a3);
      if ( v11 + 1 > *(unsigned int *)(a2 + 12) )
      {
        v31[0] = si128;
        sub_C8D5F0(a2, (const void *)(a2 + 16), v11 + 1, 0x10u, a5, v12);
        si128 = _mm_load_si128(v31);
        v13 = (__m128i *)(*(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8));
      }
      *v13 = si128;
      v17 = *(_QWORD *)a2;
      v18 = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
      *(_DWORD *)(a2 + 8) = v18;
      *(_BYTE *)(a1 + 8) = 1;
      *(_QWORD *)a1 = v17 + 16 * v18 - 16;
      *(_BYTE *)(a1 + 16) = 1;
      return a1;
    }
    v30 = (_QWORD *)(a2 + 80);
  }
  else
  {
    v14 = a3->m128i_i64[0];
    v15 = *(__m128i **)a2;
    do
    {
      if ( v15->m128i_i64[0] == v14 && v15->m128i_i64[1] == a3->m128i_i64[1] )
      {
        *(_BYTE *)(a1 + 8) = 1;
        *(_QWORD *)a1 = v15;
        *(_BYTE *)(a1 + 16) = 0;
        return a1;
      }
      ++v15;
    }
    while ( v13 != v15 );
    if ( v11 <= 3 )
      goto LABEL_13;
    v19 = *(__m128i **)a2;
    v30 = (_QWORD *)(a2 + 80);
    v31[0].m128i_i64[0] = a2 + 88;
    do
    {
      v20 = sub_34A6950(v30, v31[0].m128i_i64[0], (unsigned __int64 *)v19);
      if ( v21 )
      {
        v22 = 1;
        if ( !v20 && (_QWORD *)v31[0].m128i_i64[0] != v21 )
        {
          v27 = v21[4];
          if ( v19->m128i_i64[0] >= v27 && (v19->m128i_i64[0] != v27 || v19->m128i_i64[1] >= v21[5]) )
            v22 = 0;
        }
        v28 = v21;
        v29 = v22;
        v23 = (__m128i *)sub_22077B0(0x30u);
        v24 = (_QWORD *)v31[0].m128i_i64[0];
        v23[2] = _mm_loadu_si128(v19);
        sub_220F040(v29, (__int64)v23, v28, v24);
        ++*(_QWORD *)(a2 + 120);
      }
      ++v19;
    }
    while ( v13 != v19 );
  }
  *(_DWORD *)(a2 + 8) = 0;
  v25 = (__m128i *)sub_34A55B0((__int64)v30, (unsigned __int64 *)a3);
  if ( v26 )
    v25 = sub_349EB00((__int64)v30, (__int64)v25, v26, a3);
  *(_BYTE *)(a1 + 8) = 0;
  *(_QWORD *)a1 = v25;
  *(_BYTE *)(a1 + 16) = 1;
  return a1;
}
