// Function: sub_163BA30
// Address: 0x163ba30
//
_BYTE *__fastcall sub_163BA30(_BYTE *a1, const char *a2, __int64 a3)
{
  __int64 v3; // r14
  __int64 v6; // rax
  __m128i *v7; // rdx
  __int64 v8; // rdi
  __m128i si128; // xmm0
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // r12
  char *v13; // rsi
  __int64 v14; // rdi
  __int64 v15; // rdx
  _BYTE *v16; // rax
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // r12
  _BYTE *result; // rax

  v3 = (__int64)a2;
  v6 = sub_16E8CB0(a1, a2, a3);
  v7 = *(__m128i **)(v6 + 24);
  v8 = v6;
  if ( *(_QWORD *)(v6 + 16) - (_QWORD)v7 <= 0x27u )
  {
    a2 = "Illegal use of unrelocated value found!\n";
    sub_16E7EE0(v6, "Illegal use of unrelocated value found!\n", 40);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_42ABD70);
    v7[2].m128i_i64[0] = 0xA21646E756F6620LL;
    *v7 = si128;
    v7[1] = _mm_load_si128((const __m128i *)&xmmword_42ABD80);
    *(_QWORD *)(v6 + 24) += 40LL;
  }
  v10 = sub_16E8CB0(v8, a2, v7);
  v11 = *(_QWORD *)(v10 + 24);
  v12 = v10;
  if ( (unsigned __int64)(*(_QWORD *)(v10 + 16) - v11) <= 4 )
  {
    v12 = sub_16E7EE0(v10, "Def: ", 5);
  }
  else
  {
    *(_DWORD *)v11 = 979789124;
    *(_BYTE *)(v11 + 4) = 32;
    *(_QWORD *)(v10 + 24) += 5LL;
  }
  v13 = (char *)v12;
  v14 = v3;
  sub_155C2B0(v3, v12, 0);
  v16 = *(_BYTE **)(v12 + 24);
  if ( *(_BYTE **)(v12 + 16) == v16 )
  {
    v13 = "\n";
    v14 = v12;
    sub_16E7EE0(v12, "\n", 1);
  }
  else
  {
    *v16 = 10;
    ++*(_QWORD *)(v12 + 24);
  }
  v17 = sub_16E8CB0(v14, v13, v15);
  v18 = *(_QWORD *)(v17 + 24);
  v19 = v17;
  if ( (unsigned __int64)(*(_QWORD *)(v17 + 16) - v18) <= 4 )
  {
    v19 = sub_16E7EE0(v17, "Use: ", 5);
  }
  else
  {
    *(_DWORD *)v18 = 979727189;
    *(_BYTE *)(v18 + 4) = 32;
    *(_QWORD *)(v17 + 24) += 5LL;
  }
  sub_155C2B0(a3, v19, 0);
  result = *(_BYTE **)(v19 + 24);
  if ( *(_BYTE **)(v19 + 16) == result )
  {
    result = (_BYTE *)sub_16E7EE0(v19, "\n", 1);
  }
  else
  {
    *result = 10;
    ++*(_QWORD *)(v19 + 24);
  }
  if ( !byte_4F9EF40 )
    abort();
  *a1 = 1;
  return result;
}
