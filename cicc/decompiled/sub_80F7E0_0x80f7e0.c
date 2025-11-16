// Function: sub_80F7E0
// Address: 0x80f7e0
//
__int64 __fastcall sub_80F7E0(__int64 a1, _QWORD *a2)
{
  char v2; // al
  __int64 v3; // rbx
  _QWORD *v4; // rdi
  __int64 v5; // rax
  __int64 i; // rax
  const char *v7; // r13
  size_t v8; // rax
  _QWORD *v9; // rdi
  __int64 v10; // rax
  const char *v11; // r13
  size_t v12; // rax
  _QWORD *v13; // rdi
  __int64 result; // rax
  __m128i v15; // xmm3
  const __m128i *v16; // rax
  __m128i v17; // [rsp+0h] [rbp-40h] BYREF
  __m128i v18[3]; // [rsp+10h] [rbp-30h] BYREF

  v2 = *(_BYTE *)(a1 + 173);
  v3 = *(_QWORD *)(a1 + 128);
  if ( v2 == 4 )
  {
    v16 = *(const __m128i **)(a1 + 176);
    v17 = _mm_loadu_si128(v16);
    v18[0] = _mm_loadu_si128(v16 + 1);
  }
  else if ( v2 == 10 )
  {
    v15 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a1 + 184) + 176LL));
    v17 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a1 + 176) + 176LL));
    v18[0] = v15;
  }
  v4 = (_QWORD *)qword_4F18BE0;
  ++*a2;
  v5 = v4[2];
  if ( (unsigned __int64)(v5 + 1) > v4[1] )
  {
    sub_823810(v4);
    v4 = (_QWORD *)qword_4F18BE0;
    v5 = *(_QWORD *)(qword_4F18BE0 + 16);
  }
  *(_BYTE *)(v4[4] + v5) = 76;
  ++v4[2];
  sub_80F5E0(v3, 0, a2);
  for ( i = v3; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v7 = sub_70B4A0(*(_BYTE *)(i + 160), (__int64)&v17);
  v8 = strlen(v7);
  *a2 += v8;
  sub_8238B0(qword_4F18BE0, v7, v8);
  v9 = (_QWORD *)qword_4F18BE0;
  ++*a2;
  v10 = v9[2];
  if ( (unsigned __int64)(v10 + 1) > v9[1] )
  {
    sub_823810(v9);
    v9 = (_QWORD *)qword_4F18BE0;
    v10 = *(_QWORD *)(qword_4F18BE0 + 16);
  }
  *(_BYTE *)(v9[4] + v10) = 95;
  ++v9[2];
  for ( ; *(_BYTE *)(v3 + 140) == 12; v3 = *(_QWORD *)(v3 + 160) )
    ;
  v11 = sub_70B4A0(*(_BYTE *)(v3 + 160), (__int64)v18);
  v12 = strlen(v11);
  *a2 += v12;
  sub_8238B0(qword_4F18BE0, v11, v12);
  v13 = (_QWORD *)qword_4F18BE0;
  ++*a2;
  result = v13[2];
  if ( (unsigned __int64)(result + 1) > v13[1] )
  {
    sub_823810(v13);
    v13 = (_QWORD *)qword_4F18BE0;
    result = *(_QWORD *)(qword_4F18BE0 + 16);
  }
  *(_BYTE *)(v13[4] + result) = 69;
  ++v13[2];
  return result;
}
