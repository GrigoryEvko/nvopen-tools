// Function: sub_7F9430
// Address: 0x7f9430
//
__m128i *__fastcall sub_7F9430(__int64 a1, int a2, int a3)
{
  bool v6; // zf
  __int64 v7; // rdi
  _QWORD *v8; // rax
  __m128i *v9; // rdi
  __int64 v10; // rsi
  __m128i *result; // rax
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9

  v6 = *(_BYTE *)(a1 + 16) == 0;
  v7 = *(_QWORD *)(a1 + 8);
  if ( v6 )
  {
    v9 = (__m128i *)sub_731250(v7);
    if ( !a3 )
      goto LABEL_3;
  }
  else
  {
    v8 = sub_73E830(v7);
    v9 = (__m128i *)sub_73DCD0(v8);
    if ( !a3 )
      goto LABEL_3;
  }
  v9 = (__m128i *)sub_7F53E0((__int64)v9);
LABEL_3:
  v10 = *(_QWORD *)(a1 + 32);
  if ( *(_BYTE *)(a1 + 19) )
  {
    v10 = *(_QWORD *)(v10 + 32);
    result = sub_7E8750(v9, v10, 1);
  }
  else
  {
    result = sub_7F7100((__int64)v9, v10, a3);
  }
  if ( *(_BYTE *)(a1 + 17) && !*(_QWORD *)(a1 + 32) )
  {
    v10 = *(_QWORD *)(a1 + 48);
    result = (__m128i *)sub_73DC90(result, v10);
  }
  if ( !a2 )
    return (__m128i *)sub_731370((__int64)result, v10, v12, v13, v14, v15);
  return result;
}
