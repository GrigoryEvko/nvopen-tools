// Function: sub_2A4D550
// Address: 0x2a4d550
//
__int64 __fastcall sub_2A4D550(__int64 a1, __int64 a2, __int64 a3, const __m128i *a4)
{
  char v4; // r8
  _QWORD *v5; // r15
  __int64 v8; // rax
  __m128i v9; // xmm1
  __int64 v10; // r12
  unsigned __int64 v12; // rax
  __int8 v13; // al
  unsigned __int64 v14; // rdx
  unsigned __int64 v15; // rax
  char v16; // [rsp+Ch] [rbp-34h]

  v4 = 1;
  v5 = (_QWORD *)(a1 + 8);
  if ( !a2 && (_QWORD *)a3 != v5 )
  {
    v12 = *(_QWORD *)(a3 + 32);
    if ( a4->m128i_i64[0] >= v12 )
    {
      if ( a4->m128i_i64[0] != v12 )
      {
LABEL_6:
        v4 = 0;
        goto LABEL_2;
      }
      v13 = a4[1].m128i_i8[8];
      if ( *(_BYTE *)(a3 + 56) )
      {
        if ( !v13 )
          goto LABEL_15;
        v14 = a4->m128i_u64[1];
        v15 = *(_QWORD *)(a3 + 40);
        if ( v14 < v15 || v14 == v15 && a4[1].m128i_i64[0] < *(_QWORD *)(a3 + 48) )
          goto LABEL_15;
        if ( v14 > v15 || *(_QWORD *)(a3 + 48) < a4[1].m128i_i64[0] )
          goto LABEL_6;
      }
      else if ( v13 )
      {
        goto LABEL_6;
      }
      if ( a4[2].m128i_i64[0] >= *(_QWORD *)(a3 + 64) )
        goto LABEL_6;
    }
LABEL_15:
    v4 = 1;
  }
LABEL_2:
  v16 = v4;
  v8 = sub_22077B0(0x48u);
  v9 = _mm_loadu_si128(a4 + 1);
  v10 = v8;
  *(__m128i *)(v8 + 32) = _mm_loadu_si128(a4);
  *(__m128i *)(v8 + 48) = v9;
  *(_QWORD *)(v8 + 64) = a4[2].m128i_i64[0];
  sub_220F040(v16, v8, (_QWORD *)a3, v5);
  ++*(_QWORD *)(a1 + 40);
  return v10;
}
