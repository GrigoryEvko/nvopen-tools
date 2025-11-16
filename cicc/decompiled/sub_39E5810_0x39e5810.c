// Function: sub_39E5810
// Address: 0x39e5810
//
_BYTE *__fastcall sub_39E5810(__int64 a1, int a2, unsigned int a3, unsigned int a4, unsigned int a5)
{
  char *v8; // r15
  __int64 v9; // r12
  __m128i *v10; // rdx
  size_t v11; // r9
  _QWORD *v12; // rax
  _WORD *v13; // rax
  __int64 v14; // rax
  _WORD *v15; // rdx
  __int64 v16; // rdi
  unsigned __int64 v17; // r12
  _BYTE *result; // rax
  __int64 v19; // rdi
  __int64 v20; // r14
  char *v21; // rsi
  size_t v22; // rdx
  void *v23; // rdi
  __int64 v24; // rdi
  _WORD *v25; // rdx
  unsigned __int64 v26; // rsi
  char *v27; // rax
  char *v28; // r15
  unsigned int v29; // ecx
  unsigned int v30; // eax
  __int64 v31; // rdx
  __int64 v32; // rax

  switch ( a2 )
  {
    case 0:
    case 5:
      v8 = "bridgeos";
      break;
    case 1:
      v8 = "macos";
      break;
    case 2:
      v8 = "ios";
      break;
    case 3:
      v8 = "tvos";
      break;
    case 4:
      v8 = "watchos";
      break;
  }
  v9 = *(_QWORD *)(a1 + 272);
  v10 = *(__m128i **)(v9 + 24);
  if ( *(_QWORD *)(v9 + 16) - (_QWORD)v10 <= 0xFu )
  {
    v9 = sub_16E7EE0(*(_QWORD *)(a1 + 272), "\t.build_version ", 0x10u);
  }
  else
  {
    *v10 = _mm_load_si128((const __m128i *)&xmmword_3F7F8B0);
    *(_QWORD *)(v9 + 24) += 16LL;
  }
  v11 = strlen(v8);
  v12 = *(_QWORD **)(v9 + 24);
  if ( v11 > *(_QWORD *)(v9 + 16) - (_QWORD)v12 )
  {
    v9 = sub_16E7EE0(v9, v8, v11);
    v13 = *(_WORD **)(v9 + 24);
    goto LABEL_7;
  }
  if ( (unsigned int)v11 >= 8 )
  {
    v26 = (unsigned __int64)(v12 + 1) & 0xFFFFFFFFFFFFFFF8LL;
    *v12 = *(_QWORD *)v8;
    *(_QWORD *)((char *)v12 + (unsigned int)v11 - 8) = *(_QWORD *)&v8[(unsigned int)v11 - 8];
    v27 = (char *)v12 - v26;
    v28 = (char *)(v8 - v27);
    if ( (((_DWORD)v11 + (_DWORD)v27) & 0xFFFFFFF8) >= 8 )
    {
      v29 = (v11 + (_DWORD)v27) & 0xFFFFFFF8;
      v30 = 0;
      do
      {
        v31 = v30;
        v30 += 8;
        *(_QWORD *)(v26 + v31) = *(_QWORD *)&v28[v31];
      }
      while ( v30 < v29 );
    }
LABEL_32:
    v12 = *(_QWORD **)(v9 + 24);
    goto LABEL_33;
  }
  if ( (v11 & 4) != 0 )
  {
    *(_DWORD *)v12 = *(_DWORD *)v8;
    *(_DWORD *)((char *)v12 + (unsigned int)v11 - 4) = *(_DWORD *)&v8[(unsigned int)v11 - 4];
    v12 = *(_QWORD **)(v9 + 24);
    goto LABEL_33;
  }
  if ( (_DWORD)v11 )
  {
    *(_BYTE *)v12 = *v8;
    if ( (v11 & 2) != 0 )
    {
      *(_WORD *)((char *)v12 + (unsigned int)v11 - 2) = *(_WORD *)&v8[(unsigned int)v11 - 2];
      v12 = *(_QWORD **)(v9 + 24);
      goto LABEL_33;
    }
    goto LABEL_32;
  }
LABEL_33:
  v13 = (_WORD *)((char *)v12 + v11);
  *(_QWORD *)(v9 + 24) = v13;
LABEL_7:
  if ( *(_QWORD *)(v9 + 16) - (_QWORD)v13 <= 1u )
  {
    v9 = sub_16E7EE0(v9, ", ", 2u);
  }
  else
  {
    *v13 = 8236;
    *(_QWORD *)(v9 + 24) += 2LL;
  }
  v14 = sub_16E7A90(v9, a3);
  v15 = *(_WORD **)(v14 + 24);
  v16 = v14;
  if ( *(_QWORD *)(v14 + 16) - (_QWORD)v15 <= 1u )
  {
    v16 = sub_16E7EE0(v14, ", ", 2u);
  }
  else
  {
    *v15 = 8236;
    *(_QWORD *)(v14 + 24) += 2LL;
  }
  sub_16E7A90(v16, a4);
  if ( a5 )
  {
    v24 = *(_QWORD *)(a1 + 272);
    v25 = *(_WORD **)(v24 + 24);
    if ( *(_QWORD *)(v24 + 16) - (_QWORD)v25 <= 1u )
    {
      v32 = sub_16E7EE0(v24, ", ", 2u);
      sub_16E7A90(v32, a5);
    }
    else
    {
      *v25 = 8236;
      *(_QWORD *)(v24 + 24) += 2LL;
      sub_16E7A90(v24, a5);
    }
  }
  v17 = *(unsigned int *)(a1 + 312);
  if ( *(_DWORD *)(a1 + 312) )
  {
    v20 = *(_QWORD *)(a1 + 272);
    v21 = *(char **)(a1 + 304);
    v22 = *(unsigned int *)(a1 + 312);
    v23 = *(void **)(v20 + 24);
    if ( v17 > *(_QWORD *)(v20 + 16) - (_QWORD)v23 )
    {
      sub_16E7EE0(*(_QWORD *)(a1 + 272), v21, v22);
    }
    else
    {
      memcpy(v23, v21, v22);
      *(_QWORD *)(v20 + 24) += v17;
    }
  }
  *(_DWORD *)(a1 + 312) = 0;
  if ( (*(_BYTE *)(a1 + 680) & 1) != 0 )
    return sub_39E0440(a1);
  v19 = *(_QWORD *)(a1 + 272);
  result = *(_BYTE **)(v19 + 24);
  if ( (unsigned __int64)result >= *(_QWORD *)(v19 + 16) )
    return (_BYTE *)sub_16E7DE0(v19, 10);
  *(_QWORD *)(v19 + 24) = result + 1;
  *result = 10;
  return result;
}
