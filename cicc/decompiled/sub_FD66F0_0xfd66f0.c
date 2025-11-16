// Function: sub_FD66F0
// Address: 0xfd66f0
//
__m128i *__fastcall sub_FD66F0(unsigned __int64 a1, __int64 a2)
{
  void *v4; // rdx
  __int64 v5; // rdi
  __int64 v6; // rax
  _WORD *v7; // rdx
  __int64 v8; // rdi
  __int64 v9; // rax
  _WORD *v10; // rdx
  char *v11; // r12
  size_t v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rdi
  _QWORD *v15; // rax
  _WORD *v16; // rdx
  char v17; // al
  bool v18; // zf
  __int64 v19; // rax
  __m128i *result; // rax
  __int64 v21; // rax
  __int64 v22; // rdi
  __int64 v23; // rdi
  __int64 v24; // rax
  __m128i *v25; // rdx
  __m128i v26; // xmm0
  __int64 v27; // r12
  __int64 v28; // r14
  char v29; // bl
  unsigned __int8 *v30; // r13
  _WORD *v31; // rdx
  unsigned int v32; // ecx
  __int64 v33; // rsi
  __m128i si128; // xmm0
  __int64 v35; // r9
  __int64 v36; // r14
  char v37; // r13
  unsigned __int8 *v38; // r12
  __int64 v39; // rsi
  __int64 v40; // rdx
  __m128i *v41; // rcx
  unsigned __int64 v42; // rax
  __int64 v43; // r12
  _BYTE *v44; // rax
  __m128i v45; // xmm0
  __m128i *v46; // rdx
  __int64 v47; // [rsp+8h] [rbp-48h]
  __int64 v48[7]; // [rsp+18h] [rbp-38h] BYREF

  v4 = *(void **)(a2 + 32);
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v4 <= 0xAu )
  {
    v5 = sub_CB6200(a2, "  AliasSet[", 0xBu);
  }
  else
  {
    v5 = a2;
    qmemcpy(v4, "  AliasSet[", 11);
    *(_QWORD *)(a2 + 32) += 11LL;
  }
  v6 = sub_CB5A80(v5, a1);
  v7 = *(_WORD **)(v6 + 32);
  v8 = v6;
  if ( *(_QWORD *)(v6 + 24) - (_QWORD)v7 <= 1u )
  {
    v8 = sub_CB6200(v6, (unsigned __int8 *)", ", 2u);
  }
  else
  {
    *v7 = 8236;
    *(_QWORD *)(v6 + 32) += 2LL;
  }
  v9 = sub_CB59D0(v8, *(_DWORD *)(a1 + 64) & 0x7FFFFFF);
  v10 = *(_WORD **)(v9 + 32);
  if ( *(_QWORD *)(v9 + 24) - (_QWORD)v10 <= 1u )
  {
    sub_CB6200(v9, (unsigned __int8 *)"] ", 2u);
  }
  else
  {
    *v10 = 8285;
    *(_QWORD *)(v9 + 32) += 2LL;
  }
  v11 = "must";
  if ( (*(_BYTE *)(a1 + 67) & 0x40) != 0 )
    v11 = "may";
  v12 = strlen(v11);
  v13 = *(_QWORD *)(a2 + 32);
  if ( v12 <= *(_QWORD *)(a2 + 24) - v13 )
  {
    if ( (_DWORD)v12 )
    {
      v32 = 0;
      do
      {
        v33 = v32++;
        *(_BYTE *)(v13 + v33) = v11[v33];
      }
      while ( v32 < (unsigned int)v12 );
      v13 = *(_QWORD *)(a2 + 32);
    }
    v15 = (_QWORD *)(v12 + v13);
    v14 = a2;
    *(_QWORD *)(a2 + 32) = v15;
  }
  else
  {
    v14 = sub_CB6200(a2, (unsigned __int8 *)v11, v12);
    v15 = *(_QWORD **)(v14 + 32);
  }
  if ( *(_QWORD *)(v14 + 24) - (_QWORD)v15 <= 7u )
  {
    sub_CB6200(v14, " alias, ", 8u);
  }
  else
  {
    *v15 = 0x202C7361696C6120LL;
    *(_QWORD *)(v14 + 32) += 8LL;
  }
  v16 = *(_WORD **)(a2 + 32);
  v17 = (*(_BYTE *)(a1 + 67) >> 4) & 3;
  if ( v17 == 2 )
  {
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v16 <= 9u )
    {
      sub_CB6200(a2, "Mod       ", 0xAu);
      result = *(__m128i **)(a2 + 32);
      goto LABEL_29;
    }
    v21 = 0x2020202020646F4DLL;
LABEL_28:
    *(_QWORD *)v16 = v21;
    v16[4] = 8224;
    result = (__m128i *)(*(_QWORD *)(a2 + 32) + 10LL);
    *(_QWORD *)(a2 + 32) = result;
    goto LABEL_29;
  }
  if ( v17 == 3 )
  {
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v16 <= 9u )
    {
      sub_CB6200(a2, "Mod/Ref   ", 0xAu);
      result = *(__m128i **)(a2 + 32);
      goto LABEL_29;
    }
    v21 = 0x206665522F646F4DLL;
    goto LABEL_28;
  }
  v18 = v17 == 0;
  v19 = *(_QWORD *)(a2 + 24);
  if ( !v18 )
  {
    if ( (unsigned __int64)(v19 - (_QWORD)v16) <= 9 )
    {
      sub_CB6200(a2, "Ref       ", 0xAu);
      result = *(__m128i **)(a2 + 32);
      goto LABEL_29;
    }
    v21 = 0x2020202020666552LL;
    goto LABEL_28;
  }
  if ( (unsigned __int64)(v19 - (_QWORD)v16) <= 9 )
  {
    sub_CB6200(a2, "No access ", 0xAu);
    result = *(__m128i **)(a2 + 32);
  }
  else
  {
    qmemcpy(v16, "No access ", 10);
    result = (__m128i *)(*(_QWORD *)(a2 + 32) + 10LL);
    *(_QWORD *)(a2 + 32) = result;
  }
LABEL_29:
  if ( *(_QWORD *)(a1 + 16) )
  {
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)result <= 0xEu )
    {
      v22 = sub_CB6200(a2, " forwarding to ", 0xFu);
    }
    else
    {
      v22 = a2;
      qmemcpy(result, " forwarding to ", 15);
      *(_QWORD *)(a2 + 32) += 15LL;
    }
    sub_CB5A80(v22, *(_QWORD *)(a1 + 16));
    result = *(__m128i **)(a2 + 32);
  }
  if ( *(_DWORD *)(a1 + 32) )
  {
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)result <= 0x11u )
    {
      sub_CB6200(a2, "Memory locations: ", 0x12u);
      result = *(__m128i **)(a2 + 32);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F8C2A0);
      result[1].m128i_i16[0] = 8250;
      *result = si128;
      result = (__m128i *)(*(_QWORD *)(a2 + 32) + 18LL);
      *(_QWORD *)(a2 + 32) = result;
    }
    v35 = *(_QWORD *)(a1 + 24);
    v47 = v35 + 48LL * *(unsigned int *)(a1 + 32);
    if ( v35 != v47 )
    {
      v36 = *(_QWORD *)(a1 + 24);
      v37 = 1;
      while ( !v37 )
      {
        if ( *(_QWORD *)(a2 + 24) - (_QWORD)result > 1u )
        {
          result->m128i_i16[0] = 8236;
          result = (__m128i *)(*(_QWORD *)(a2 + 32) + 2LL);
          *(_QWORD *)(a2 + 32) = result;
LABEL_67:
          v38 = *(unsigned __int8 **)v36;
          if ( *(__m128i **)(a2 + 24) == result )
            goto LABEL_80;
          goto LABEL_68;
        }
        sub_CB6200(a2, (unsigned __int8 *)", ", 2u);
        result = *(__m128i **)(a2 + 32);
        v38 = *(unsigned __int8 **)v36;
        if ( *(__m128i **)(a2 + 24) == result )
        {
LABEL_80:
          v39 = sub_CB6200(a2, (unsigned __int8 *)"(", 1u);
          goto LABEL_69;
        }
LABEL_68:
        result->m128i_i8[0] = 40;
        v39 = a2;
        ++*(_QWORD *)(a2 + 32);
LABEL_69:
        sub_A5BF40(v38, v39, 1, 0);
        v40 = *(_QWORD *)(v36 + 8);
        if ( v40 == 0xBFFFFFFFFFFFFFFELL )
        {
          v46 = *(__m128i **)(a2 + 32);
          if ( *(_QWORD *)(a2 + 24) - (_QWORD)v46 <= 0xFu )
          {
            sub_CB6200(a2, ", unknown after)", 0x10u);
            result = *(__m128i **)(a2 + 32);
          }
          else
          {
            *v46 = _mm_load_si128((const __m128i *)&xmmword_3F8C2B0);
            result = (__m128i *)(*(_QWORD *)(a2 + 32) + 16LL);
            *(_QWORD *)(a2 + 32) = result;
          }
        }
        else
        {
          v41 = *(__m128i **)(a2 + 32);
          v42 = *(_QWORD *)(a2 + 24) - (_QWORD)v41;
          if ( v40 == -1 )
          {
            if ( v42 <= 0x19 )
            {
              sub_CB6200(a2, ", unknown before-or-after)", 0x1Au);
              result = *(__m128i **)(a2 + 32);
            }
            else
            {
              v45 = _mm_load_si128((const __m128i *)&xmmword_3F8C2C0);
              qmemcpy(&v41[1], "-or-after)", 10);
              *v41 = v45;
              result = (__m128i *)(*(_QWORD *)(a2 + 32) + 26LL);
              *(_QWORD *)(a2 + 32) = result;
            }
          }
          else
          {
            if ( v42 <= 1 )
            {
              v43 = sub_CB6200(a2, (unsigned __int8 *)", ", 2u);
            }
            else
            {
              v43 = a2;
              v41->m128i_i16[0] = 8236;
              *(_QWORD *)(a2 + 32) += 2LL;
            }
            v48[0] = *(_QWORD *)(v36 + 8);
            sub_D66290(v48, v43);
            v44 = *(_BYTE **)(v43 + 32);
            if ( *(_BYTE **)(v43 + 24) == v44 )
            {
              sub_CB6200(v43, (unsigned __int8 *)")", 1u);
            }
            else
            {
              *v44 = 41;
              ++*(_QWORD *)(v43 + 32);
            }
            result = *(__m128i **)(a2 + 32);
          }
        }
        v36 += 48;
        if ( v47 == v36 )
          goto LABEL_34;
      }
      v37 = 0;
      goto LABEL_67;
    }
  }
LABEL_34:
  if ( *(_QWORD *)(a1 + 48) != *(_QWORD *)(a1 + 40) )
  {
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)result <= 4u )
    {
      v23 = sub_CB6200(a2, "\n    ", 5u);
    }
    else
    {
      result->m128i_i32[0] = 538976266;
      v23 = a2;
      result->m128i_i8[4] = 32;
      *(_QWORD *)(a2 + 32) += 5LL;
    }
    v24 = sub_CB59D0(v23, 0xAAAAAAAAAAAAAAABLL * ((__int64)(*(_QWORD *)(a1 + 48) - *(_QWORD *)(a1 + 40)) >> 3));
    v25 = *(__m128i **)(v24 + 32);
    if ( *(_QWORD *)(v24 + 24) - (_QWORD)v25 <= 0x16u )
    {
      sub_CB6200(v24, " Unknown instructions: ", 0x17u);
    }
    else
    {
      v26 = _mm_load_si128((const __m128i *)&xmmword_3F8C2D0);
      v25[1].m128i_i32[0] = 1852795252;
      v25[1].m128i_i16[2] = 14963;
      v25[1].m128i_i8[6] = 32;
      *v25 = v26;
      *(_QWORD *)(v24 + 32) += 23LL;
    }
    v27 = *(_QWORD *)(a1 + 40);
    v28 = *(_QWORD *)(a1 + 48);
    v29 = 1;
    if ( v28 == v27 )
    {
LABEL_48:
      result = *(__m128i **)(a2 + 32);
      goto LABEL_49;
    }
    while ( 1 )
    {
      v30 = *(unsigned __int8 **)(v27 + 16);
      if ( v29 )
        break;
      v31 = *(_WORD **)(a2 + 32);
      if ( *(_QWORD *)(a2 + 24) - (_QWORD)v31 > 1u )
      {
        *v31 = 8236;
        *(_QWORD *)(a2 + 32) += 2LL;
LABEL_42:
        if ( (v30[7] & 0x10) != 0 )
          goto LABEL_43;
LABEL_47:
        v27 += 24;
        sub_A69870((__int64)v30, (_BYTE *)a2, 0);
        if ( v28 == v27 )
          goto LABEL_48;
      }
      else
      {
        sub_CB6200(a2, (unsigned __int8 *)", ", 2u);
        if ( (v30[7] & 0x10) == 0 )
          goto LABEL_47;
LABEL_43:
        sub_A5BF40(v30, a2, 1, 0);
        v27 += 24;
        if ( v28 == v27 )
          goto LABEL_48;
      }
    }
    v29 = 0;
    goto LABEL_42;
  }
LABEL_49:
  if ( result == *(__m128i **)(a2 + 24) )
    return (__m128i *)sub_CB6200(a2, (unsigned __int8 *)"\n", 1u);
  result->m128i_i8[0] = 10;
  ++*(_QWORD *)(a2 + 32);
  return result;
}
