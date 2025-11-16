// Function: sub_7EDD70
// Address: 0x7edd70
//
__m128i **__fastcall sub_7EDD70(__m128i *a1, __m128i **a2)
{
  __m128i *v2; // r12
  __m128i *v3; // r15
  __int64 v4; // r13
  const __m128i *v5; // rdi
  _QWORD *v7; // rsi
  __int64 i; // rdx
  __int8 v9; // dl
  const __m128i *v10; // rdi
  int v11; // edx
  const __m128i *v12; // [rsp+0h] [rbp-70h]
  int v14; // [rsp+14h] [rbp-5Ch] BYREF
  const __m128i *v15; // [rsp+18h] [rbp-58h] BYREF
  _BYTE v16[80]; // [rsp+20h] [rbp-50h] BYREF

  if ( a1 )
  {
    v2 = a1;
    while ( 1 )
    {
      v3 = v2;
      v2 = (__m128i *)v2[1].m128i_i64[0];
      v15 = v3;
      *(_QWORD *)dword_4D03F38 = v3->m128i_i64[0];
      *(_QWORD *)dword_4F07508 = *(_QWORD *)dword_4D03F38;
      v4 = qword_4D03F68[4];
      if ( !v4 || *(__m128i **)(v4 + 16) != v3 )
      {
        sub_7EC960(v3);
        goto LABEL_6;
      }
      if ( !dword_4D044B4 )
      {
        sub_7EC960(v3);
        goto LABEL_18;
      }
      v9 = v3[2].m128i_i8[8];
      v14 = 0;
      if ( v9 == 7 )
        break;
      if ( v9 == 15 )
      {
        v11 = *(_BYTE *)(v3[5].m128i_i64[0] + 64) & 1;
LABEL_27:
        v10 = v3;
        if ( !v11 )
          goto LABEL_25;
      }
      sub_7E70C0(&v15, &v14, (__int64)v16);
      v10 = v15;
LABEL_25:
      sub_7EC960(v10);
LABEL_18:
      sub_7E1720(*(_QWORD *)(v4 + 16), (__int64)v16);
      sub_7E9190(v4, (__int64)v16);
      v7 = qword_4D03F68;
      qword_4D03F68[2] = v4;
      qword_4F06BC0 = v4;
      i = 0;
      v7[5] = 0;
      if ( (*(_BYTE *)(v4 + 1) & 1) != 0 )
      {
        for ( i = *(_QWORD *)(v4 + 48); *(_BYTE *)i != 2; i = *(_QWORD *)(i + 56) )
          ;
      }
      v7[4] = i;
LABEL_6:
      v5 = (const __m128i *)v3[1].m128i_i64[0];
      if ( v3[2].m128i_i8[8] != 11 || v3[4].m128i_i64[1] || *(_DWORD *)v3[5].m128i_i64[0] )
        goto LABEL_9;
      if ( v2 != v5 )
      {
        v12 = (const __m128i *)v5[1].m128i_i64[0];
        sub_732B40(v5, v3);
        v3[1].m128i_i64[0] = (__int64)v12;
        v5 = v12;
LABEL_9:
        while ( v2 != v5 )
        {
          v3 = (__m128i *)v5;
          v5 = (const __m128i *)v5[1].m128i_i64[0];
        }
      }
      if ( !v2 )
        goto LABEL_11;
    }
    v11 = *(_BYTE *)(v3[4].m128i_i64[1] + 120) & 1;
    goto LABEL_27;
  }
  v3 = 0;
LABEL_11:
  *a2 = v3;
  return a2;
}
