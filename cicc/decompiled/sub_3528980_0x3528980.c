// Function: sub_3528980
// Address: 0x3528980
//
void __fastcall sub_3528980(
        __int64 a1,
        unsigned __int64 *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        char a8,
        char a9)
{
  void (*v11)(); // rax
  __int64 v12; // rcx
  __int64 *v13; // rbx
  __int64 v14; // r12
  unsigned __int64 v15; // rsi
  __int64 v16; // rdx
  __int64 v17; // rbx
  __int64 *i; // r13
  __int64 v19; // r14
  __int64 v20; // rax
  __int64 v21; // rax
  const __m128i *v22; // rax
  __int64 *v23; // rbx
  __int64 v24; // r13
  __int64 v25; // rdx
  char v26; // [rsp+4h] [rbp-5Ch]
  __int64 v28; // [rsp+10h] [rbp-50h]
  __int64 v29; // [rsp+18h] [rbp-48h]
  __int64 v30; // [rsp+28h] [rbp-38h]

  v28 = a3;
  v26 = a9;
  v29 = a5;
  v11 = *(void (**)())(*(_QWORD *)a7 + 616LL);
  if ( v11 != nullsub_1683 )
    ((void (__fastcall *)(__int64, unsigned __int64 *, char *, __int64))v11)(a7, a2, &a8, a3);
  v12 = *(_QWORD *)v28;
  v13 = *(__int64 **)v28;
  v30 = *(_QWORD *)v28 + 8LL * *(unsigned int *)(v28 + 8);
  if ( *(_QWORD *)v28 != v30 )
  {
    do
    {
      v14 = *v13++;
      sub_2E31040((__int64 *)(a1 + 40), v14);
      v15 = *a2;
      v16 = *(_QWORD *)v14;
      *(_QWORD *)(v14 + 8) = a2;
      v15 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)v14 = v15 | v16 & 7;
      *(_QWORD *)(v15 + 8) = v14;
      a3 = *a2 & 7;
      *a2 = a3 | v14;
    }
    while ( (__int64 *)v30 != v13 );
  }
  v17 = *(_QWORD *)a4 + 8LL * *(unsigned int *)(a4 + 8);
  for ( i = *(__int64 **)a4; (__int64 *)v17 != i; ++i )
  {
    v19 = *i;
    sub_2E88E20(*i);
    v12 = *(_QWORD *)a6;
    a3 = *(_QWORD *)a6;
    v20 = 24LL * *(unsigned int *)(a6 + 8);
    if ( v20 )
    {
      do
      {
        if ( *(_QWORD *)(a3 + 8) == v19 )
        {
          v22 = (const __m128i *)(v12 + v20 - 24);
          if ( v22 != (const __m128i *)a3 )
          {
            *(__m128i *)a3 = _mm_loadu_si128(v22);
            *(_DWORD *)(a3 + 16) = v22[1].m128i_i32[0];
            *(_BYTE *)(*(_QWORD *)(a6 + 208) + *(unsigned int *)(*(_QWORD *)a6 + 24LL * *(unsigned int *)(a6 + 8) - 24)) = -85 * ((a3 - *(_QWORD *)a6) >> 3);
            v12 = *(_QWORD *)a6;
          }
          v21 = (unsigned int)(*(_DWORD *)(a6 + 8) - 1);
          *(_DWORD *)(a6 + 8) = v21;
        }
        else
        {
          v21 = *(unsigned int *)(a6 + 8);
          a3 += 24;
        }
        v20 = 24 * v21;
        a5 = v12 + v20;
      }
      while ( a3 != v12 + v20 );
    }
  }
  if ( v26 )
  {
    v23 = *(__int64 **)v28;
    v24 = *(_QWORD *)v28 + 8LL * *(unsigned int *)(v28 + 8);
    if ( v24 != *(_QWORD *)v28 )
    {
      do
      {
        v25 = *v23++;
        sub_2EEC4F0(v29, a1, v25, a6);
      }
      while ( (__int64 *)v24 != v23 );
    }
  }
  else
  {
    sub_2EE8CA0(v29, a1, a3, v12, a5, a6);
  }
}
