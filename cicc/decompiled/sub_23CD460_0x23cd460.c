// Function: sub_23CD460
// Address: 0x23cd460
//
int __fastcall sub_23CD460(__int64 a1, __m128i *a2, __int64 a3)
{
  unsigned int v4; // eax
  void (__fastcall *v5)(__m128i *, __m128i *, __int64); // rax
  __int64 v6; // rdx
  __m128i v7; // xmm1
  __m128i v8; // xmm0
  __int64 v9; // rax
  int v10; // r15d
  __m128i v12; // [rsp+0h] [rbp-60h] BYREF
  void (__fastcall *v13)(__m128i *, __m128i *, __int64); // [rsp+10h] [rbp-50h]
  __int64 v14; // [rsp+18h] [rbp-48h]
  __int64 v15; // [rsp+20h] [rbp-40h]

  if ( &_pthread_key_create )
  {
    v4 = pthread_mutex_lock((pthread_mutex_t *)(a1 + 176));
    if ( v4 )
      sub_4264C5(v4);
  }
  v5 = (void (__fastcall *)(__m128i *, __m128i *, __int64))a2[1].m128i_i64[0];
  v6 = v14;
  a2[1].m128i_i64[0] = 0;
  v7 = _mm_loadu_si128(&v12);
  v8 = _mm_loadu_si128(a2);
  v15 = a3;
  v13 = v5;
  v9 = a2[1].m128i_i64[1];
  a2[1].m128i_i64[1] = v6;
  *a2 = v7;
  v14 = v9;
  v12 = v8;
  sub_2265350((unsigned __int64 *)(a1 + 96), &v12);
  if ( v13 )
    v13(&v12, &v12, 3);
  v10 = *(_DWORD *)(a1 + 312)
      + -858993459 * ((__int64)(*(_QWORD *)(a1 + 128) - *(_QWORD *)(a1 + 112)) >> 3)
      + -858993459 * ((__int64)(*(_QWORD *)(a1 + 144) - *(_QWORD *)(a1 + 152)) >> 3)
      + 4 * (3 * ((__int64)(*(_QWORD *)(a1 + 168) - *(_QWORD *)(a1 + 136)) >> 3) - 3);
  if ( &_pthread_key_create )
    pthread_mutex_unlock((pthread_mutex_t *)(a1 + 176));
  sub_2210B50((pthread_cond_t *)(a1 + 216));
  return sub_23CCA50(a1, v10);
}
