// Function: sub_1239130
// Address: 0x1239130
//
__int64 __fastcall sub_1239130(__int64 a1, __int64 *a2, _QWORD *a3, __int32 a4)
{
  __int64 v4; // r15
  __int64 v10; // rax
  unsigned int v11; // edi
  __int64 v12; // r8
  __int64 v13; // rsi
  __m128i *v14; // r8
  __int64 v15; // rax
  __int64 v16; // [rsp+8h] [rbp-58h]
  int v17; // [rsp+1Ch] [rbp-44h] BYREF
  __m128i v18[4]; // [rsp+20h] [rbp-40h] BYREF

  v4 = a1 + 176;
  *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
  if ( (unsigned __int8)sub_120AFE0(a1, 16, "expected ':' here")
    || (unsigned __int8)sub_120AFE0(a1, 12, "expected '(' here") )
  {
    return 1;
  }
  if ( *(_DWORD *)(a1 + 240) == 506 )
  {
    *a2 = 0;
    v10 = a3[2];
    v11 = *(_DWORD *)(a1 + 280);
    v12 = *(_QWORD *)(a1 + 232);
    v13 = (__int64)(a3 + 1);
    v17 = *(_DWORD *)(a1 + 280);
    if ( !v10 )
      goto LABEL_26;
    do
    {
      if ( v11 > *(_DWORD *)(v10 + 32) )
      {
        v10 = *(_QWORD *)(v10 + 24);
      }
      else
      {
        v13 = v10;
        v10 = *(_QWORD *)(v10 + 16);
      }
    }
    while ( v10 );
    if ( (_QWORD *)v13 == a3 + 1 || v11 < *(_DWORD *)(v13 + 32) )
    {
LABEL_26:
      v16 = v12;
      v18[0].m128i_i64[0] = (__int64)&v17;
      v15 = sub_1239060(a3, v13, (unsigned int **)v18);
      v12 = v16;
      v13 = v15;
    }
    v18[0].m128i_i32[0] = a4;
    v18[0].m128i_i64[1] = v12;
    v14 = *(__m128i **)(v13 + 48);
    if ( v14 == *(__m128i **)(v13 + 56) )
    {
      sub_12171B0((const __m128i **)(v13 + 40), *(const __m128i **)(v13 + 48), v18);
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
    *(_DWORD *)(a1 + 240) = sub_1205200(v4);
  }
  else if ( (unsigned __int8)sub_120AFE0(a1, 412, "expected 'guid' here")
         || (unsigned __int8)sub_120AFE0(a1, 16, "expected ':' here")
         || (unsigned __int8)sub_120C050(a1, a2) )
  {
    return 1;
  }
  if ( (unsigned __int8)sub_120AFE0(a1, 4, "expected ',' here")
    || (unsigned __int8)sub_120AFE0(a1, 459, "expected 'offset' here")
    || (unsigned __int8)sub_120AFE0(a1, 16, "expected ':' here")
    || (unsigned __int8)sub_120C050(a1, a2 + 1) )
  {
    return 1;
  }
  return sub_120AFE0(a1, 13, "expected ')' here");
}
