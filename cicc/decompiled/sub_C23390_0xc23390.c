// Function: sub_C23390
// Address: 0xc23390
//
__int64 __fastcall sub_C23390(__int64 a1, __int64 a2, char a3)
{
  __int64 result; // rax
  unsigned __int64 v4; // rsi
  __int64 v5; // rax
  __int64 v6; // rdx
  unsigned __int64 v7; // rbx
  __m128i *v8; // rsi
  unsigned __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // r15
  __m128i *v14; // rsi
  __int64 v15; // rax
  __m128i *v16; // rdx
  __int64 v17; // rdx
  __int64 v18; // rsi
  unsigned __int64 v19; // rax
  __int64 v20; // rax
  __m128i v21; // [rsp+0h] [rbp-80h] BYREF
  __m128i v22; // [rsp+10h] [rbp-70h] BYREF
  char v23; // [rsp+20h] [rbp-60h]
  unsigned __int64 v24; // [rsp+30h] [rbp-50h] BYREF
  char v25; // [rsp+40h] [rbp-40h]

  if ( a3 )
  {
    if ( !(_BYTE)a2 )
    {
      v15 = sub_CB72A0(a1, a2);
      v16 = *(__m128i **)(v15 + 32);
      if ( *(_QWORD *)(v15 + 24) - (_QWORD)v16 <= 0x2Fu )
      {
        sub_CB6200(v15, "If FixedLengthMD5 is true, UseMD5 has to be true", 48);
      }
      else
      {
        *v16 = _mm_load_si128((const __m128i *)&xmmword_3F64F20);
        v16[1] = _mm_load_si128((const __m128i *)&xmmword_3F64F30);
        v16[2] = _mm_load_si128((const __m128i *)&xmmword_3F64F40);
        *(_QWORD *)(v15 + 32) += 48LL;
      }
    }
    sub_C21E40((__int64)&v24, (_QWORD *)a1);
    if ( (v25 & 1) == 0 || (result = (unsigned int)v24, !(_DWORD)v24) )
    {
      v4 = v24;
      if ( *(_QWORD *)(a1 + 216) < *(_QWORD *)(a1 + 208) + 8 * v24 )
      {
        sub_C1AFD0();
        return 4;
      }
      else
      {
        v5 = *(_QWORD *)(a1 + 224);
        if ( *(_QWORD *)(a1 + 232) != v5 )
          *(_QWORD *)(a1 + 232) = v5;
        sub_C22490((const __m128i **)(a1 + 224), v4);
        v6 = *(_QWORD *)(a1 + 208);
        if ( v24 )
        {
          v7 = 0;
          do
          {
            while ( 1 )
            {
              v10 = *(_QWORD *)(v6 + 8 * v7);
              v8 = *(__m128i **)(a1 + 232);
              v22.m128i_i64[0] = 0;
              v22.m128i_i64[1] = v10;
              if ( v8 != *(__m128i **)(a1 + 240) )
                break;
              ++v7;
              sub_C22DD0((const __m128i **)(a1 + 224), v8, &v22);
              v9 = v24;
              v6 = *(_QWORD *)(a1 + 208);
              if ( v24 <= v7 )
                goto LABEL_15;
            }
            if ( v8 )
            {
              *v8 = _mm_loadu_si128(&v22);
              v8 = *(__m128i **)(a1 + 232);
              v6 = *(_QWORD *)(a1 + 208);
            }
            v9 = v24;
            ++v7;
            *(_QWORD *)(a1 + 232) = v8 + 1;
          }
          while ( v9 > v7 );
LABEL_15:
          v11 = v6 + 8 * v9;
        }
        else
        {
          v11 = *(_QWORD *)(a1 + 208);
        }
        if ( !*(_BYTE *)(a1 + 178) )
          *(_QWORD *)(a1 + 296) = v6;
        *(_QWORD *)(a1 + 208) = v11;
        sub_C1AFD0();
        return 0;
      }
    }
    return result;
  }
  if ( !(_BYTE)a2 )
    return sub_C22F50(a1);
  sub_C21E40((__int64)&v22, (_QWORD *)a1);
  if ( (v23 & 1) == 0 || (result = v22.m128i_u32[0], !v22.m128i_i32[0]) )
  {
    v12 = *(_QWORD *)(a1 + 224);
    if ( v12 != *(_QWORD *)(a1 + 232) )
      *(_QWORD *)(a1 + 232) = v12;
    sub_C22490((const __m128i **)(a1 + 224), v22.m128i_u64[0]);
    if ( *(_BYTE *)(a1 + 178) )
    {
      if ( v22.m128i_i64[0] )
        goto LABEL_25;
LABEL_46:
      sub_C1AFD0();
      return 0;
    }
    v17 = *(_QWORD *)(a1 + 272);
    v18 = v22.m128i_i64[0];
    v19 = (*(_QWORD *)(a1 + 280) - v17) >> 3;
    if ( v22.m128i_i64[0] > v19 )
    {
      sub_C22AA0(a1 + 272, v22.m128i_i64[0] - v19);
      if ( !v22.m128i_i64[0] )
      {
LABEL_43:
        if ( *(_BYTE *)(a1 + 178) )
          goto LABEL_46;
        v17 = *(_QWORD *)(a1 + 272);
LABEL_45:
        *(_QWORD *)(a1 + 296) = v17;
        goto LABEL_46;
      }
    }
    else
    {
      if ( v22.m128i_i64[0] < v19 )
      {
        v20 = v17 + 8 * v22.m128i_i64[0];
        if ( *(_QWORD *)(a1 + 280) != v20 )
          *(_QWORD *)(a1 + 280) = v20;
      }
      if ( !v18 )
        goto LABEL_45;
    }
LABEL_25:
    v13 = 0;
    while ( 1 )
    {
      sub_C21E40((__int64)&v24, (_QWORD *)a1);
      if ( (v25 & 1) != 0 )
      {
        result = (unsigned int)v24;
        if ( (_DWORD)v24 )
          break;
      }
      if ( !*(_BYTE *)(a1 + 178) )
        *(_QWORD *)(*(_QWORD *)(a1 + 272) + 8 * v13) = v24;
      v21.m128i_i64[0] = 0;
      v14 = *(__m128i **)(a1 + 232);
      v21.m128i_i64[1] = v24;
      if ( v14 == *(__m128i **)(a1 + 240) )
      {
        sub_C22DD0((const __m128i **)(a1 + 224), v14, &v21);
      }
      else
      {
        if ( v14 )
        {
          *v14 = _mm_loadu_si128(&v21);
          v14 = *(__m128i **)(a1 + 232);
        }
        *(_QWORD *)(a1 + 232) = v14 + 1;
      }
      if ( v22.m128i_i64[0] <= (unsigned __int64)++v13 )
        goto LABEL_43;
    }
  }
  return result;
}
