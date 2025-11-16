// Function: sub_ED0A20
// Address: 0xed0a20
//
__int64 __fastcall sub_ED0A20(__int64 a1, __int64 a2)
{
  unsigned __int8 v2; // al
  __int64 v3; // rax
  __int64 v4; // rdx
  _BYTE *v5; // rsi
  __m128i *v6; // rdx
  __int64 v7; // rax
  __int64 v9[2]; // [rsp+0h] [rbp-30h] BYREF
  __m128i v10[2]; // [rsp+10h] [rbp-20h] BYREF

  if ( a2 )
  {
    v2 = *(_BYTE *)(a2 - 16);
    if ( (v2 & 2) != 0 )
    {
      v3 = sub_B91420(**(_QWORD **)(a2 - 32));
      v5 = (_BYTE *)v3;
      if ( v3 )
      {
LABEL_4:
        v9[0] = (__int64)v10;
        sub_ED0450(v9, v5, v3 + v4);
        v6 = (__m128i *)v9[0];
        v7 = v9[1];
        *(_QWORD *)a1 = a1 + 16;
        if ( v6 != v10 )
        {
          *(_QWORD *)a1 = v6;
          *(_QWORD *)(a1 + 16) = v10[0].m128i_i64[0];
LABEL_6:
          *(_QWORD *)(a1 + 8) = v7;
          *(_BYTE *)(a1 + 32) = 1;
          return a1;
        }
LABEL_9:
        *(__m128i *)(a1 + 16) = _mm_load_si128(v10);
        goto LABEL_6;
      }
    }
    else
    {
      v3 = sub_B91420(*(_QWORD *)(a2 - 8LL * ((v2 >> 2) & 0xF) - 16));
      v5 = (_BYTE *)v3;
      if ( v3 )
        goto LABEL_4;
    }
    v10[0].m128i_i8[0] = 0;
    *(_QWORD *)a1 = a1 + 16;
    v7 = 0;
    goto LABEL_9;
  }
  *(_QWORD *)(a1 + 32) = 0;
  *(_OWORD *)a1 = 0;
  *(_OWORD *)(a1 + 16) = 0;
  return a1;
}
