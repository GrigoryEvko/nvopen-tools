// Function: sub_1C56DA0
// Address: 0x1c56da0
//
__int64 __fastcall sub_1C56DA0(_QWORD *a1, const __m128i *a2)
{
  const __m128i *v2; // rbp
  _QWORD *v3; // rax
  unsigned __int64 v5; // rdx
  _QWORD *v6; // rsi
  const __m128i *v8[2]; // [rsp-10h] [rbp-10h] BYREF

  v3 = (_QWORD *)a1[2];
  if ( !v3 )
  {
    v6 = a1 + 1;
LABEL_12:
    v8[1] = v2;
    v8[0] = a2;
    return sub_1C56CE0(a1, v6, v8) + 48;
  }
  v5 = a2->m128i_i64[0];
  v6 = a1 + 1;
  do
  {
    while ( v3[4] >= v5 && (v3[4] != v5 || v3[5] >= a2->m128i_i64[1]) )
    {
      v6 = v3;
      v3 = (_QWORD *)v3[2];
      if ( !v3 )
        goto LABEL_8;
    }
    v3 = (_QWORD *)v3[3];
  }
  while ( v3 );
LABEL_8:
  if ( a1 + 1 == v6 || v6[4] > v5 || v6[4] == v5 && a2->m128i_i64[1] < v6[5] )
    goto LABEL_12;
  return (__int64)(v6 + 6);
}
