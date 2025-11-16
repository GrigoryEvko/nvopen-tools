// Function: sub_C26F80
// Address: 0xc26f80
//
__int64 __fastcall sub_C26F80(__int64 a1)
{
  __int64 result; // rax
  __m128i *v3; // rsi
  __m128i *v4; // rdi
  int v5; // r12d
  unsigned int v6; // eax
  __int64 v7; // rax
  unsigned __int64 v8; // rdx
  unsigned __int64 v9; // rcx
  unsigned __int64 v10; // rdx
  _BYTE *v11; // rsi
  unsigned __int64 v12; // rax
  __int64 v13; // rax
  __m128i *v14; // rdx
  __int64 v15; // rdi
  __m128i si128; // xmm0
  __int64 v17; // rdi
  _BYTE *v18; // rax
  __m128i v19; // rax
  unsigned __int64 v20; // rax
  unsigned int v21; // [rsp+4h] [rbp-6Ch]
  char v22; // [rsp+1Bh] [rbp-55h] BYREF
  unsigned int v23; // [rsp+1Ch] [rbp-54h] BYREF
  __m128i v24; // [rsp+20h] [rbp-50h] BYREF
  __m128i v25[4]; // [rsp+30h] [rbp-40h] BYREF

  result = sub_C219E0(a1, 2852126720LL);
  v21 = result;
  if ( !(_DWORD)result )
  {
    v3 = (__m128i *)&v23;
    v4 = (__m128i *)(a1 + 208);
    if ( (unsigned __int8)sub_C1FF60((__int64)v4, &v23) )
    {
      v5 = 0;
      if ( !v23 )
      {
LABEL_35:
        sub_C1AFD0();
        return v21;
      }
      while ( (unsigned __int64)(*(_QWORD *)(a1 + 232) + 4LL) <= *(_QWORD *)(a1 + 216) )
      {
        v6 = sub_C5F610(a1 + 208, a1 + 232, a1 + 240);
        if ( !v6 )
          goto LABEL_24;
        if ( *(int *)(a1 + 256) <= 4 )
        {
          v19.m128i_i64[0] = sub_C5ED50(a1 + 208, a1 + 232, 4 * v6, a1 + 240);
          v24 = v19;
          v22 = 0;
          v20 = sub_C931B0(&v24, &v22, 1, 0);
          v11 = (_BYTE *)v24.m128i_i64[0];
          v10 = v20;
          if ( v20 == -1 )
          {
            v10 = v24.m128i_u64[1];
          }
          else if ( v24.m128i_i64[1] <= v20 )
          {
            v10 = v24.m128i_u64[1];
          }
        }
        else
        {
          v7 = sub_C5ED50(a1 + 208, a1 + 232, v6, a1 + 240);
          v9 = v8;
          v10 = v8 - 1;
          v11 = (_BYTE *)v7;
          if ( v9 < v10 )
            v10 = 0;
        }
        v12 = *(_QWORD *)(a1 + 240) & 0xFFFFFFFFFFFFFFFELL | ((*(_QWORD *)(a1 + 240) & 0xFFFFFFFFFFFFFFFELL) != 0);
        *(_QWORD *)(a1 + 240) = v12;
        if ( (v12 & 0xFFFFFFFFFFFFFFFELL) != 0 )
          goto LABEL_24;
        v24.m128i_i64[0] = (__int64)v25;
        sub_C1EB20(v24.m128i_i64, v11, (__int64)&v11[v10]);
        v3 = *(__m128i **)(a1 + 272);
        if ( v3 == *(__m128i **)(a1 + 280) )
        {
          sub_8F99A0((__m128i **)(a1 + 264), v3, &v24);
          v4 = (__m128i *)v24.m128i_i64[0];
        }
        else
        {
          v4 = (__m128i *)v24.m128i_i64[0];
          if ( v3 )
          {
            v3->m128i_i64[0] = (__int64)v3[1].m128i_i64;
            if ( (__m128i *)v24.m128i_i64[0] == v25 )
            {
              v3[1] = _mm_load_si128(v25);
            }
            else
            {
              v3->m128i_i64[0] = v24.m128i_i64[0];
              v3[1].m128i_i64[0] = v25[0].m128i_i64[0];
            }
            v3->m128i_i64[1] = v24.m128i_i64[1];
            v3 = *(__m128i **)(a1 + 272);
            v24.m128i_i64[1] = 0;
            v25[0].m128i_i8[0] = 0;
            v24.m128i_i64[0] = (__int64)v25;
            v4 = v25;
          }
          v3 += 2;
          *(_QWORD *)(a1 + 272) = v3;
        }
        if ( v4 != v25 )
        {
          v3 = (__m128i *)(v25[0].m128i_i64[0] + 1);
          j_j___libc_free_0(v4, v25[0].m128i_i64[0] + 1);
        }
        if ( v23 <= ++v5 )
          goto LABEL_35;
      }
      v13 = sub_CB72A0(v4, v3);
      v14 = *(__m128i **)(v13 + 32);
      v15 = v13;
      if ( *(_QWORD *)(v13 + 24) - (_QWORD)v14 <= 0x20u )
      {
        v15 = sub_CB6200(v13, "unexpected end of memory buffer: ", 33);
      }
      else
      {
        si128 = _mm_load_si128((const __m128i *)&xmmword_3F64EF0);
        v14[2].m128i_i8[0] = 32;
        *v14 = si128;
        v14[1] = _mm_load_si128((const __m128i *)&xmmword_3F64F00);
        *(_QWORD *)(v13 + 32) += 33LL;
      }
      v17 = sub_CB59D0(v15, *(_QWORD *)(a1 + 232));
      v18 = *(_BYTE **)(v17 + 32);
      if ( *(_BYTE **)(v17 + 24) == v18 )
      {
        sub_CB6200(v17, "\n", 1);
      }
      else
      {
        *v18 = 10;
        ++*(_QWORD *)(v17 + 32);
      }
    }
LABEL_24:
    sub_C1AFD0();
    return 4;
  }
  return result;
}
