// Function: sub_12BDA30
// Address: 0x12bda30
//
__int64 __fastcall sub_12BDA30(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 v5; // rdi
  _BYTE *v6; // rax
  __int64 v7; // rax
  __m128i *v8; // rdx
  __int64 v9; // rdi
  __m128i si128; // xmm0
  __int64 v11; // rdi
  _BYTE *v12; // rax
  __int64 v13; // rdi
  _BYTE *v14; // rax
  _QWORD v16[4]; // [rsp+0h] [rbp-50h] BYREF
  int v17; // [rsp+20h] [rbp-30h]
  __int64 v18; // [rsp+28h] [rbp-28h]

  v18 = a1 + 80;
  v17 = 1;
  memset(&v16[1], 0, 24);
  v16[0] = &unk_49EFBE0;
  v4 = sub_16E7EE0(v16, "IR version ", 11);
  v5 = sub_16E7A90(v4, a2);
  v6 = *(_BYTE **)(v5 + 24);
  if ( *(_BYTE **)(v5 + 16) == v6 )
  {
    v5 = sub_16E7EE0(v5, ".", 1);
  }
  else
  {
    *v6 = 46;
    ++*(_QWORD *)(v5 + 24);
  }
  v7 = sub_16E7A90(v5, a3);
  v8 = *(__m128i **)(v7 + 24);
  v9 = v7;
  if ( *(_QWORD *)(v7 + 16) - (_QWORD)v8 <= 0x22u )
  {
    v9 = sub_16E7EE0(v7, " incompatible with current version ", 35);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_4281860);
    v8[2].m128i_i8[2] = 32;
    v8[2].m128i_i16[0] = 28271;
    *v8 = si128;
    v8[1] = _mm_load_si128((const __m128i *)&xmmword_4281870);
    *(_QWORD *)(v7 + 24) += 35LL;
  }
  v11 = sub_16E7AB0(v9, 2);
  v12 = *(_BYTE **)(v11 + 24);
  if ( *(_BYTE **)(v11 + 16) == v12 )
  {
    v11 = sub_16E7EE0(v11, ".", 1);
  }
  else
  {
    *v12 = 46;
    ++*(_QWORD *)(v11 + 24);
  }
  v13 = sub_16E7AB0(v11, 0);
  v14 = *(_BYTE **)(v13 + 24);
  if ( *(_BYTE **)(v13 + 16) == v14 )
  {
    sub_16E7EE0(v13, "\n", 1);
  }
  else
  {
    *v14 = 10;
    ++*(_QWORD *)(v13 + 24);
  }
  sub_16E7BC0(v16);
  return 0;
}
