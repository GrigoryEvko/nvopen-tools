// Function: sub_34CDA70
// Address: 0x34cda70
//
bool __fastcall sub_34CDA70(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rbx
  __int64 v5; // r13
  __int64 (*v6)(); // rax
  __int64 v7; // rdi
  __m128i v8; // xmm0
  __m128i v9; // xmm1
  __int64 *v10; // rdx
  __m128i *v11; // rax
  __int64 v12; // rcx
  _BOOL4 v13; // eax
  __m128i v15; // [rsp+0h] [rbp-50h] BYREF
  __m128i v16; // [rsp+10h] [rbp-40h]
  __int64 v17; // [rsp+20h] [rbp-30h]

  v4 = 0;
  v5 = *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8LL);
  v6 = *(__int64 (**)())(*(_QWORD *)v5 + 16LL);
  if ( v6 != sub_23CE270 )
  {
    v4 = ((__int64 (__fastcall *)(_QWORD))v6)(*(_QWORD *)(*(_QWORD *)(a1 + 32) + 8LL));
    v6 = *(__int64 (**)())(*(_QWORD *)v5 + 16LL);
  }
  v7 = 0;
  if ( v6 != sub_23CE270 )
    v7 = ((__int64 (__fastcall *)(__int64, __int64))v6)(v5, a3);
  v8 = _mm_loadu_si128((const __m128i *)(v4 + 232));
  v9 = _mm_loadu_si128((const __m128i *)(v4 + 248));
  v10 = (__int64 *)(v7 + 232);
  v17 = *(_QWORD *)(v4 + 264);
  v11 = &v15;
  v15 = v8;
  v16 = v9;
  do
  {
    v12 = *v10++;
    v11->m128i_i64[0] &= v12;
    v11 = (__m128i *)((char *)v11 + 8);
  }
  while ( (__int64 *)(v7 + 272) != v10 );
  v13 = *(_OWORD *)(v7 + 232) != *(_OWORD *)&v15
     || *(_OWORD *)(v7 + 248) != *(_OWORD *)&v16
     || v17 != *(_QWORD *)(v7 + 264);
  return !v13;
}
