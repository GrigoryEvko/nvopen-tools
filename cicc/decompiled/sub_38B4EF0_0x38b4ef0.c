// Function: sub_38B4EF0
// Address: 0x38b4ef0
//
__int64 __fastcall sub_38B4EF0(__int64 a1, __int64 *a2, _QWORD *a3, __int32 a4)
{
  __int64 v4; // r15
  unsigned int v10; // edi
  __int64 v11; // rax
  __int64 v12; // r8
  __int64 v13; // rsi
  __m128i *v14; // r8
  __int64 v15; // rax
  __int64 v16; // [rsp+8h] [rbp-58h]
  int v17; // [rsp+1Ch] [rbp-44h] BYREF
  __m128i v18[4]; // [rsp+20h] [rbp-40h] BYREF

  v4 = a1 + 8;
  *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
  if ( (unsigned __int8)sub_388AF10(a1, 16, "expected ':' here")
    || (unsigned __int8)sub_388AF10(a1, 12, "expected '(' here") )
  {
    return 1;
  }
  if ( *(_DWORD *)(a1 + 64) == 371 )
  {
    *a2 = 0;
    v10 = *(_DWORD *)(a1 + 104);
    v11 = a3[2];
    v12 = *(_QWORD *)(a1 + 56);
    v13 = (__int64)(a3 + 1);
    v17 = *(_DWORD *)(a1 + 104);
    if ( !v11 )
      goto LABEL_26;
    do
    {
      if ( v10 > *(_DWORD *)(v11 + 32) )
      {
        v11 = *(_QWORD *)(v11 + 24);
      }
      else
      {
        v13 = v11;
        v11 = *(_QWORD *)(v11 + 16);
      }
    }
    while ( v11 );
    if ( (_QWORD *)v13 == a3 + 1 || v10 < *(_DWORD *)(v13 + 32) )
    {
LABEL_26:
      v16 = v12;
      v18[0].m128i_i64[0] = (__int64)&v17;
      v15 = sub_38B4270(a3, v13, (unsigned int **)v18);
      v12 = v16;
      v13 = v15;
    }
    v18[0].m128i_i32[0] = a4;
    v18[0].m128i_i64[1] = v12;
    v14 = *(__m128i **)(v13 + 48);
    if ( v14 == *(__m128i **)(v13 + 56) )
    {
      sub_3894FE0((unsigned __int64 *)(v13 + 40), *(const __m128i **)(v13 + 48), v18);
    }
    else
    {
      if ( v14 )
      {
        *v14 = _mm_loadu_si128(v18);
        v14 = *(__m128i **)(v13 + 48);
      }
      *(_QWORD *)(v13 + 48) = v14 + 1;
    }
    *(_DWORD *)(a1 + 64) = sub_3887100(v4);
  }
  else if ( (unsigned __int8)sub_388AF10(a1, 306, "expected 'guid' here")
         || (unsigned __int8)sub_388AF10(a1, 16, "expected ':' here")
         || (unsigned __int8)sub_388BD80(a1, a2) )
  {
    return 1;
  }
  if ( (unsigned __int8)sub_388AF10(a1, 4, "expected ',' here")
    || (unsigned __int8)sub_388AF10(a1, 338, "expected 'offset' here")
    || (unsigned __int8)sub_388AF10(a1, 16, "expected ':' here")
    || (unsigned __int8)sub_388BD80(a1, a2 + 1) )
  {
    return 1;
  }
  return sub_388AF10(a1, 13, "expected ')' here");
}
