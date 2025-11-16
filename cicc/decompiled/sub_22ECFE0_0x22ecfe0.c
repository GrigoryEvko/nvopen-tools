// Function: sub_22ECFE0
// Address: 0x22ecfe0
//
_BYTE *__fastcall sub_22ECFE0(_BYTE *a1, __int64 a2, __int64 a3)
{
  _QWORD *v4; // rax
  __m128i *v5; // rdx
  __m128i si128; // xmm0
  _QWORD *v7; // rax
  __int64 v8; // rdx
  __int64 v9; // r12
  _BYTE *v10; // rax
  _QWORD *v11; // rax
  __int64 v12; // rdx
  __int64 v13; // r12
  _BYTE *result; // rax

  v4 = sub_CB72A0();
  v5 = (__m128i *)v4[4];
  if ( v4[3] - (_QWORD)v5 <= 0x27u )
  {
    sub_CB6200((__int64)v4, "Illegal use of unrelocated value found!\n", 0x28u);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_42ABD70);
    v5[2].m128i_i64[0] = 0xA21646E756F6620LL;
    *v5 = si128;
    v5[1] = _mm_load_si128((const __m128i *)&xmmword_42ABD80);
    v4[4] += 40LL;
  }
  v7 = sub_CB72A0();
  v8 = v7[4];
  v9 = (__int64)v7;
  if ( (unsigned __int64)(v7[3] - v8) <= 4 )
  {
    v9 = sub_CB6200((__int64)v7, (unsigned __int8 *)"Def: ", 5u);
  }
  else
  {
    *(_DWORD *)v8 = 979789124;
    *(_BYTE *)(v8 + 4) = 32;
    v7[4] += 5LL;
  }
  sub_A69870(a2, (_BYTE *)v9, 0);
  v10 = *(_BYTE **)(v9 + 32);
  if ( *(_BYTE **)(v9 + 24) == v10 )
  {
    sub_CB6200(v9, (unsigned __int8 *)"\n", 1u);
  }
  else
  {
    *v10 = 10;
    ++*(_QWORD *)(v9 + 32);
  }
  v11 = sub_CB72A0();
  v12 = v11[4];
  v13 = (__int64)v11;
  if ( (unsigned __int64)(v11[3] - v12) <= 4 )
  {
    v13 = sub_CB6200((__int64)v11, (unsigned __int8 *)"Use: ", 5u);
  }
  else
  {
    *(_DWORD *)v12 = 979727189;
    *(_BYTE *)(v12 + 4) = 32;
    v11[4] += 5LL;
  }
  sub_A69870(a3, (_BYTE *)v13, 0);
  result = *(_BYTE **)(v13 + 32);
  if ( *(_BYTE **)(v13 + 24) == result )
  {
    result = (_BYTE *)sub_CB6200(v13, (unsigned __int8 *)"\n", 1u);
  }
  else
  {
    *result = 10;
    ++*(_QWORD *)(v13 + 32);
  }
  if ( !byte_4FDC228 )
    abort();
  *a1 = 1;
  return result;
}
