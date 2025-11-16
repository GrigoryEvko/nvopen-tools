// Function: sub_1B2AAB0
// Address: 0x1b2aab0
//
unsigned __int64 __fastcall sub_1B2AAB0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rdx
  unsigned __int64 result; // rax
  __int64 v6; // rdi
  int v7; // ecx
  unsigned int v8; // edx
  __int64 v9; // r8
  __int64 v10; // rbx
  __m128i *v11; // rdx
  __int64 v12; // r13
  __m128i si128; // xmm0
  __m128i *v14; // rdx
  __m128i v15; // xmm0
  __int64 v16; // rdi
  __int64 v17; // rax
  void *v18; // rdx
  __int64 v19; // r13
  __int64 v20; // rdi
  _QWORD *v21; // rdx
  _BYTE *v22; // rax
  _DWORD *v23; // rdx
  __m128i *v24; // rdx
  __m128i v25; // xmm0
  _QWORD *v26; // rdx
  __m128i *v27; // rdx
  __m128i v28; // xmm0
  void *v29; // rdx
  __int64 v30; // rdx
  unsigned int v31; // r9d
  __int64 v32; // rax

  v4 = *(_QWORD *)(a1 + 8);
  result = *(unsigned int *)(v4 + 104);
  if ( !(_DWORD)result )
    return result;
  v6 = *(_QWORD *)(v4 + 88);
  v7 = result - 1;
  v8 = (result - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  result = v6 + 16LL * v8;
  v9 = *(_QWORD *)result;
  if ( a2 == *(_QWORD *)result )
  {
LABEL_3:
    v10 = *(_QWORD *)(result + 8);
    if ( !v10 )
      return result;
    v11 = *(__m128i **)(a3 + 24);
    v12 = a3;
    if ( *(_QWORD *)(a3 + 16) - (_QWORD)v11 <= 0x14u )
    {
      sub_16E7EE0(a3, "; Has predicate info\n", 0x15u);
      result = *(unsigned int *)(v10 + 24);
      if ( !(_DWORD)result )
        goto LABEL_6;
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_42C7750);
      v11[1].m128i_i32[0] = 1868983913;
      v11[1].m128i_i8[4] = 10;
      *v11 = si128;
      *(_QWORD *)(a3 + 24) += 21LL;
      result = *(unsigned int *)(v10 + 24);
      if ( !(_DWORD)result )
      {
LABEL_6:
        v14 = *(__m128i **)(a3 + 24);
        if ( *(_QWORD *)(a3 + 16) - (_QWORD)v14 <= 0x23u )
        {
          v16 = sub_16E7EE0(a3, "; branch predicate info { TrueEdge: ", 0x24u);
        }
        else
        {
          v15 = _mm_load_si128((const __m128i *)&xmmword_42C7760);
          v14[2].m128i_i32[0] = 540697959;
          v16 = a3;
          *v14 = v15;
          v14[1] = _mm_load_si128((const __m128i *)&xmmword_42C7770);
          *(_QWORD *)(a3 + 24) += 36LL;
        }
        v17 = sub_16E7AB0(v16, *(unsigned __int8 *)(v10 + 64));
        v18 = *(void **)(v17 + 24);
        v19 = v17;
        if ( *(_QWORD *)(v17 + 16) - (_QWORD)v18 <= 0xBu )
        {
          v19 = sub_16E7EE0(v17, " Comparison:", 0xCu);
        }
        else
        {
          qmemcpy(v18, " Comparison:", 12);
          *(_QWORD *)(v17 + 24) += 12LL;
        }
        v20 = *(_QWORD *)(v10 + 40);
LABEL_11:
        sub_155C2B0(v20, v19, 0);
        v21 = *(_QWORD **)(v19 + 24);
        if ( *(_QWORD *)(v19 + 16) - (_QWORD)v21 <= 7u )
        {
          sub_16E7EE0(v19, " Edge: [", 8u);
        }
        else
        {
          *v21 = 0x5B203A6567644520LL;
          *(_QWORD *)(v19 + 24) += 8LL;
        }
        sub_15537D0(*(_QWORD *)(v10 + 48), a3, 1, 0);
        v22 = *(_BYTE **)(a3 + 24);
        if ( *(_BYTE **)(a3 + 16) == v22 )
        {
          sub_16E7EE0(a3, ",", 1u);
        }
        else
        {
          *v22 = 44;
          ++*(_QWORD *)(a3 + 24);
        }
        sub_15537D0(*(_QWORD *)(v10 + 56), a3, 1, 0);
        v23 = *(_DWORD **)(a3 + 24);
        result = *(_QWORD *)(a3 + 16) - (_QWORD)v23;
        if ( result <= 3 )
          return sub_16E7EE0(a3, "] }\n", 4u);
        *v23 = 175972445;
        *(_QWORD *)(a3 + 24) += 4LL;
        return result;
      }
    }
    if ( (_DWORD)result == 2 )
    {
      v24 = *(__m128i **)(a3 + 24);
      if ( *(_QWORD *)(a3 + 16) - (_QWORD)v24 <= 0x24u )
      {
        v19 = sub_16E7EE0(a3, "; switch predicate info { CaseValue: ", 0x25u);
      }
      else
      {
        v25 = _mm_load_si128((const __m128i *)&xmmword_42C7780);
        v24[2].m128i_i32[0] = 979727724;
        v19 = a3;
        v24[2].m128i_i8[4] = 32;
        *v24 = v25;
        v24[1] = _mm_load_si128((const __m128i *)&xmmword_42C7790);
        *(_QWORD *)(a3 + 24) += 37LL;
      }
      sub_155C2B0(*(_QWORD *)(v10 + 64), v19, 0);
      v26 = *(_QWORD **)(v19 + 24);
      if ( *(_QWORD *)(v19 + 16) - (_QWORD)v26 <= 7u )
      {
        v19 = sub_16E7EE0(v19, " Switch:", 8u);
      }
      else
      {
        *v26 = 0x3A68637469775320LL;
        *(_QWORD *)(v19 + 24) += 8LL;
      }
      v20 = *(_QWORD *)(v10 + 72);
      goto LABEL_11;
    }
    if ( (_DWORD)result == 1 )
    {
      v27 = *(__m128i **)(a3 + 24);
      if ( *(_QWORD *)(a3 + 16) - (_QWORD)v27 <= 0x18u )
      {
        v32 = sub_16E7EE0(a3, "; assume predicate info {", 0x19u);
        v29 = *(void **)(v32 + 24);
        v12 = v32;
      }
      else
      {
        v28 = _mm_load_si128((const __m128i *)&xmmword_42C77A0);
        v27[1].m128i_i8[8] = 123;
        v27[1].m128i_i64[0] = 0x206F666E69206574LL;
        *v27 = v28;
        v29 = (void *)(*(_QWORD *)(a3 + 24) + 25LL);
        *(_QWORD *)(a3 + 24) = v29;
      }
      if ( *(_QWORD *)(v12 + 16) - (_QWORD)v29 <= 0xBu )
      {
        v12 = sub_16E7EE0(v12, " Comparison:", 0xCu);
      }
      else
      {
        qmemcpy(v29, " Comparison:", 12);
        *(_QWORD *)(v12 + 24) += 12LL;
      }
      sub_155C2B0(*(_QWORD *)(v10 + 40), v12, 0);
      v30 = *(_QWORD *)(v12 + 24);
      if ( (unsigned __int64)(*(_QWORD *)(v12 + 16) - v30) <= 2 )
      {
        return sub_16E7EE0(v12, " }\n", 3u);
      }
      else
      {
        *(_BYTE *)(v30 + 2) = 10;
        *(_WORD *)v30 = 32032;
        *(_QWORD *)(v12 + 24) += 3LL;
        return 32032;
      }
    }
  }
  else
  {
    result = 1;
    while ( v9 != -8 )
    {
      v31 = result + 1;
      v8 = v7 & (result + v8);
      result = v6 + 16LL * v8;
      v9 = *(_QWORD *)result;
      if ( a2 == *(_QWORD *)result )
        goto LABEL_3;
      result = v31;
    }
  }
  return result;
}
