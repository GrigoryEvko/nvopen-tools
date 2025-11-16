// Function: sub_73C240
// Address: 0x73c240
//
__m128i *__fastcall sub_73C240(__m128i *a1)
{
  __m128i *result; // rax
  __m128i *i; // r12
  __int64 v3; // rdx
  _QWORD *v4; // rcx
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // r13
  __int64 v8; // rax
  char v9; // cl
  char v10; // dl
  const __m128i *v11; // r14
  __int64 v12; // rax
  __int64 v13; // r15
  __m128i *v14; // r13
  __int8 v15; // dl
  __m128i *v16; // [rsp+8h] [rbp-38h]
  __int64 v17; // [rsp+8h] [rbp-38h]
  __m128i *v18; // [rsp+8h] [rbp-38h]

  if ( !dword_4F06978 || !dword_4D048B8 )
    return a1;
  for ( i = a1; i[8].m128i_i8[12] == 12; i = (__m128i *)i[10].m128i_i64[0] )
    ;
  if ( !(unsigned int)sub_8D76D0(i) )
    return a1;
  v3 = i[7].m128i_i64[1];
  v4 = 0;
  if ( !v3 )
    goto LABEL_15;
  if ( *(_BYTE *)(v3 + 16) != 6 )
  {
    do
    {
      v5 = *(_QWORD *)v3;
      v4 = (_QWORD *)v3;
      if ( !*(_QWORD *)v3 )
        goto LABEL_15;
      v3 = *(_QWORD *)v3;
    }
    while ( *(_BYTE *)(v5 + 16) != 6 );
  }
  result = *(__m128i **)(v3 + 8);
  if ( v4 )
  {
    *v4 = *(_QWORD *)v3;
    *(_QWORD *)v3 = i[7].m128i_i64[1];
    i[7].m128i_i64[1] = v3;
  }
  if ( !result )
  {
LABEL_15:
    if ( (unsigned int)sub_8D3D10(i) )
    {
      v11 = (const __m128i *)sub_8D4870(i);
      v12 = v11[10].m128i_i64[1];
      v13 = *(_QWORD *)(v12 + 56);
      *(_QWORD *)(v12 + 56) = 0;
      v14 = (__m128i *)sub_7259C0(v11[8].m128i_i8[12]);
      sub_73C230(v11, v14);
      *(_QWORD *)(v11[10].m128i_i64[1] + 56) = v13;
      v15 = a1[-1].m128i_i8[8];
      v14[9].m128i_i64[1] = 0;
      v14[-1].m128i_i8[8] = v15 & 8 | v14[-1].m128i_i8[8] & 0xF7;
      v18 = (__m128i *)sub_7259C0(i[8].m128i_i8[12]);
      sub_73C230(i, v18);
      v8 = (__int64)v18;
      v18[10].m128i_i64[1] = (__int64)v14;
    }
    else
    {
      v6 = i[10].m128i_i64[1];
      v7 = *(_QWORD *)(v6 + 56);
      *(_QWORD *)(v6 + 56) = 0;
      v16 = (__m128i *)sub_7259C0(i[8].m128i_i8[12]);
      sub_73C230(i, v16);
      v8 = (__int64)v16;
      *(_QWORD *)(i[10].m128i_i64[1] + 56) = v7;
    }
    v9 = a1[-1].m128i_i8[8] & 8;
    v10 = *(_BYTE *)(v8 - 8);
    *(_QWORD *)(v8 + 152) = 0;
    *(_BYTE *)(v8 - 8) = v9 | v10 & 0xF7;
    v17 = v8;
    sub_728520(i, 6, v8);
    return (__m128i *)v17;
  }
  return result;
}
