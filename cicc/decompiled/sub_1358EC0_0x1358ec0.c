// Function: sub_1358EC0
// Address: 0x1358ec0
//
_BYTE *__fastcall sub_1358EC0(__int64 a1, __int64 a2)
{
  void *v4; // rdx
  __int64 v5; // rdi
  __int64 v6; // rax
  _WORD *v7; // rdx
  __int64 v8; // rdi
  __int64 v9; // rax
  _WORD *v10; // rdx
  const char *v11; // r14
  size_t v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rdi
  _QWORD *v15; // rax
  _WORD *v16; // rdx
  char v17; // al
  bool v18; // zf
  __int64 v19; // rax
  _BYTE *result; // rax
  __int64 v21; // rax
  __int64 v22; // rdi
  __int64 *v23; // rbx
  __int64 v24; // r14
  __int64 v25; // rsi
  _WORD *v26; // rdx
  __int64 v27; // rdi
  __int64 v28; // rdi
  _BYTE *v29; // rax
  unsigned int v30; // ecx
  __int64 v31; // rsi
  __int64 v32; // rdi
  __int64 v33; // rax
  __m128i *v34; // rdx
  __m128i si128; // xmm0
  __int64 v36; // rax
  unsigned __int64 v37; // rdx
  __int64 v38; // r14
  __int64 i; // rbx
  _WORD *v40; // rdx
  __int64 v41; // rdi

  v4 = *(void **)(a2 + 24);
  if ( *(_QWORD *)(a2 + 16) - (_QWORD)v4 <= 0xAu )
  {
    v5 = sub_16E7EE0(a2, "  AliasSet[", 11);
  }
  else
  {
    v5 = a2;
    qmemcpy(v4, "  AliasSet[", 11);
    *(_QWORD *)(a2 + 24) += 11LL;
  }
  v6 = sub_16E7B40(v5, a1);
  v7 = *(_WORD **)(v6 + 24);
  v8 = v6;
  if ( *(_QWORD *)(v6 + 16) - (_QWORD)v7 <= 1u )
  {
    v8 = sub_16E7EE0(v6, ", ", 2);
  }
  else
  {
    *v7 = 8236;
    *(_QWORD *)(v6 + 24) += 2LL;
  }
  v9 = sub_16E7A90(v8, *(_DWORD *)(a1 + 64) & 0x7FFFFFF);
  v10 = *(_WORD **)(v9 + 24);
  if ( *(_QWORD *)(v9 + 16) - (_QWORD)v10 <= 1u )
  {
    sub_16E7EE0(v9, "] ", 2);
  }
  else
  {
    *v10 = 8285;
    *(_QWORD *)(v9 + 24) += 2LL;
  }
  v11 = "must";
  if ( (*(_BYTE *)(a1 + 67) & 0x40) != 0 )
    v11 = "may";
  v12 = strlen(v11);
  v13 = *(_QWORD *)(a2 + 24);
  if ( v12 <= *(_QWORD *)(a2 + 16) - v13 )
  {
    if ( (_DWORD)v12 )
    {
      v30 = 0;
      do
      {
        v31 = v30++;
        *(_BYTE *)(v13 + v31) = v11[v31];
      }
      while ( v30 < (unsigned int)v12 );
      v13 = *(_QWORD *)(a2 + 24);
    }
    v15 = (_QWORD *)(v12 + v13);
    v14 = a2;
    *(_QWORD *)(a2 + 24) = v15;
  }
  else
  {
    v14 = sub_16E7EE0(a2, v11);
    v15 = *(_QWORD **)(v14 + 24);
  }
  if ( *(_QWORD *)(v14 + 16) - (_QWORD)v15 <= 7u )
  {
    sub_16E7EE0(v14, " alias, ", 8);
  }
  else
  {
    *v15 = 0x202C7361696C6120LL;
    *(_QWORD *)(v14 + 24) += 8LL;
  }
  v16 = *(_WORD **)(a2 + 24);
  v17 = (*(_BYTE *)(a1 + 67) >> 4) & 3;
  if ( v17 == 2 )
  {
    if ( *(_QWORD *)(a2 + 16) - (_QWORD)v16 <= 9u )
    {
      sub_16E7EE0(a2, "Mod       ", 10);
      result = *(_BYTE **)(a2 + 24);
      goto LABEL_29;
    }
    v21 = 0x2020202020646F4DLL;
    goto LABEL_28;
  }
  if ( v17 == 3 )
  {
    if ( *(_QWORD *)(a2 + 16) - (_QWORD)v16 <= 9u )
    {
      sub_16E7EE0(a2, "Mod/Ref   ", 10);
      result = *(_BYTE **)(a2 + 24);
      goto LABEL_29;
    }
    v21 = 0x206665522F646F4DLL;
    goto LABEL_28;
  }
  v18 = v17 == 0;
  v19 = *(_QWORD *)(a2 + 16);
  if ( !v18 )
  {
    if ( (unsigned __int64)(v19 - (_QWORD)v16) <= 9 )
    {
      sub_16E7EE0(a2, "Ref       ", 10);
      result = *(_BYTE **)(a2 + 24);
      goto LABEL_29;
    }
    v21 = 0x2020202020666552LL;
LABEL_28:
    *(_QWORD *)v16 = v21;
    v16[4] = 8224;
    result = (_BYTE *)(*(_QWORD *)(a2 + 24) + 10LL);
    *(_QWORD *)(a2 + 24) = result;
    goto LABEL_29;
  }
  if ( (unsigned __int64)(v19 - (_QWORD)v16) <= 9 )
  {
    sub_16E7EE0(a2, "No access ", 10);
    result = *(_BYTE **)(a2 + 24);
  }
  else
  {
    qmemcpy(v16, "No access ", 10);
    result = (_BYTE *)(*(_QWORD *)(a2 + 24) + 10LL);
    *(_QWORD *)(a2 + 24) = result;
  }
LABEL_29:
  if ( *(char *)(a1 + 67) < 0 )
  {
    if ( *(_QWORD *)(a2 + 16) - (_QWORD)result <= 0xAu )
    {
      sub_16E7EE0(a2, "[volatile] ", 11);
      result = *(_BYTE **)(a2 + 24);
    }
    else
    {
      qmemcpy(result, "[volatile] ", 11);
      result = (_BYTE *)(*(_QWORD *)(a2 + 24) + 11LL);
      *(_QWORD *)(a2 + 24) = result;
    }
  }
  if ( *(_QWORD *)(a1 + 32) )
  {
    if ( *(_QWORD *)(a2 + 16) - (_QWORD)result <= 0xEu )
    {
      v22 = sub_16E7EE0(a2, " forwarding to ", 15);
    }
    else
    {
      v22 = a2;
      qmemcpy(result, " forwarding to ", 15);
      *(_QWORD *)(a2 + 24) += 15LL;
    }
    sub_16E7B40(v22, *(_QWORD *)(a1 + 32));
    result = *(_BYTE **)(a2 + 24);
  }
  if ( *(_QWORD *)(a1 + 16) )
  {
    if ( *(_QWORD *)(a2 + 16) - (_QWORD)result <= 9u )
    {
      sub_16E7EE0(a2, "Pointers: ", 10);
      result = *(_BYTE **)(a2 + 24);
    }
    else
    {
      qmemcpy(result, "Pointers: ", 10);
      result = (_BYTE *)(*(_QWORD *)(a2 + 24) + 10LL);
      *(_QWORD *)(a2 + 24) = result;
    }
    v23 = *(__int64 **)(a1 + 16);
    if ( v23 )
    {
      while ( 1 )
      {
LABEL_38:
        v24 = *v23;
        if ( *(_BYTE **)(a2 + 16) != result )
        {
LABEL_39:
          *result = 40;
          v25 = a2;
          ++*(_QWORD *)(a2 + 24);
          goto LABEL_40;
        }
        while ( 1 )
        {
          v25 = sub_16E7EE0(a2, "(", 1);
LABEL_40:
          sub_15537D0(v24, v25, 1);
          v26 = *(_WORD **)(a2 + 24);
          if ( *(_QWORD *)(a2 + 16) - (_QWORD)v26 <= 1u )
          {
            v27 = sub_16E7EE0(a2, ", ", 2);
          }
          else
          {
            v27 = a2;
            *v26 = 8236;
            *(_QWORD *)(a2 + 24) += 2LL;
          }
          v28 = sub_16E7A90(v27, v23[4]);
          v29 = *(_BYTE **)(v28 + 24);
          if ( *(_BYTE **)(v28 + 16) == v29 )
          {
            sub_16E7EE0(v28, ")", 1);
            v23 = (__int64 *)v23[2];
            result = *(_BYTE **)(a2 + 24);
            if ( !v23 )
              goto LABEL_58;
          }
          else
          {
            *v29 = 41;
            ++*(_QWORD *)(v28 + 24);
            v23 = (__int64 *)v23[2];
            result = *(_BYTE **)(a2 + 24);
            if ( !v23 )
              goto LABEL_58;
          }
          if ( *(__int64 **)(a1 + 16) == v23 )
            goto LABEL_38;
          if ( *(_QWORD *)(a2 + 16) - (_QWORD)result <= 1u )
            break;
          *(_WORD *)result = 8236;
          result = (_BYTE *)(*(_QWORD *)(a2 + 24) + 2LL);
          *(_QWORD *)(a2 + 24) = result;
          v24 = *v23;
          if ( *(_BYTE **)(a2 + 16) != result )
            goto LABEL_39;
        }
        sub_16E7EE0(a2, ", ", 2);
        result = *(_BYTE **)(a2 + 24);
      }
    }
  }
LABEL_58:
  if ( *(_QWORD *)(a1 + 48) != *(_QWORD *)(a1 + 40) )
  {
    if ( *(_QWORD *)(a2 + 16) - (_QWORD)result <= 4u )
    {
      v32 = sub_16E7EE0(a2, "\n    ", 5);
    }
    else
    {
      *(_DWORD *)result = 538976266;
      v32 = a2;
      result[4] = 32;
      *(_QWORD *)(a2 + 24) += 5LL;
    }
    v33 = sub_16E7A90(v32, 0xAAAAAAAAAAAAAAABLL * ((__int64)(*(_QWORD *)(a1 + 48) - *(_QWORD *)(a1 + 40)) >> 3));
    v34 = *(__m128i **)(v33 + 24);
    if ( *(_QWORD *)(v33 + 16) - (_QWORD)v34 <= 0x16u )
    {
      sub_16E7EE0(v33, " Unknown instructions: ", 23);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F8C2D0);
      v34[1].m128i_i32[0] = 1852795252;
      v34[1].m128i_i16[2] = 14963;
      v34[1].m128i_i8[6] = 32;
      *v34 = si128;
      *(_QWORD *)(v33 + 24) += 23LL;
    }
    v36 = *(_QWORD *)(a1 + 40);
    v37 = 0xAAAAAAAAAAAAAAABLL * ((*(_QWORD *)(a1 + 48) - v36) >> 3);
    if ( (_DWORD)v37 )
    {
      v38 = (unsigned int)(v37 - 1);
      for ( i = 0; ; ++i )
      {
        v41 = *(_QWORD *)(v36 + 24 * i + 16);
        if ( !v41 )
          goto LABEL_66;
        if ( (*(_BYTE *)(v41 + 23) & 0x20) != 0 )
          break;
        sub_155C2B0(v41, a2, 0);
        if ( v38 == i )
          goto LABEL_74;
LABEL_67:
        if ( (_DWORD)i != -1 )
        {
          v40 = *(_WORD **)(a2 + 24);
          if ( *(_QWORD *)(a2 + 16) - (_QWORD)v40 <= 1u )
          {
            sub_16E7EE0(a2, ", ", 2);
          }
          else
          {
            *v40 = 8236;
            *(_QWORD *)(a2 + 24) += 2LL;
          }
        }
        v36 = *(_QWORD *)(a1 + 40);
      }
      sub_15537D0(v41, a2, 1);
LABEL_66:
      if ( v38 == i )
        goto LABEL_74;
      goto LABEL_67;
    }
LABEL_74:
    result = *(_BYTE **)(a2 + 24);
  }
  if ( result == *(_BYTE **)(a2 + 16) )
    return (_BYTE *)sub_16E7EE0(a2, "\n", 1);
  *result = 10;
  ++*(_QWORD *)(a2 + 24);
  return result;
}
