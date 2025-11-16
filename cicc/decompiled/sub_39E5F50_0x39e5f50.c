// Function: sub_39E5F50
// Address: 0x39e5f50
//
_BYTE *__fastcall sub_39E5F50(__int64 a1, int a2, __int64 a3)
{
  size_t v5; // r14
  char *v6; // r15
  __int64 v7; // rdi
  _BYTE *v8; // rax
  _DWORD *v9; // rdx
  _BYTE *v10; // rax
  _BYTE *v11; // rax
  _BYTE *v12; // rdx
  _BYTE **v13; // rbx
  __int64 v14; // r13
  _BYTE *v15; // r15
  _BYTE **i; // rbx
  __int64 v17; // rdi
  _WORD *v18; // rdx
  unsigned __int64 v19; // r13
  _BYTE *result; // rax
  __int64 v21; // rdi
  __int64 v22; // r14
  char *v23; // rsi
  size_t v24; // rdx
  void *v25; // rdi
  __int64 v26; // rax
  _BYTE *v27; // rdx
  __int64 v28; // rax
  unsigned __int64 v29; // rsi
  _BYTE *v30; // rax
  char *v31; // r15
  unsigned int v32; // edx
  unsigned int v33; // eax
  __int64 v34; // rcx

  switch ( a2 )
  {
    case 1:
      v5 = 8;
      v6 = "AdrpAdrp";
      break;
    case 2:
      v5 = 7;
      v6 = "AdrpLdr";
      break;
    case 3:
      v5 = 10;
      v6 = "AdrpAddLdr";
      break;
    case 4:
      v5 = 13;
      v6 = "AdrpLdrGotLdr";
      break;
    case 5:
      v5 = 10;
      v6 = "AdrpAddStr";
      break;
    case 6:
      v5 = 13;
      v6 = "AdrpLdrGotStr";
      break;
    case 7:
      v5 = 7;
      v6 = "AdrpAdd";
      break;
    case 8:
      v5 = 10;
      v6 = "AdrpLdrGot";
      break;
    default:
      v5 = 0;
      v6 = 0;
      break;
  }
  v7 = *(_QWORD *)(a1 + 272);
  v8 = *(_BYTE **)(v7 + 24);
  if ( *(_BYTE **)(v7 + 16) == v8 )
  {
    v28 = sub_16E7EE0(v7, "\t", 1u);
    v9 = *(_DWORD **)(v28 + 24);
    v7 = v28;
  }
  else
  {
    *v8 = 9;
    v9 = (_DWORD *)(*(_QWORD *)(v7 + 24) + 1LL);
    *(_QWORD *)(v7 + 24) = v9;
  }
  if ( *(_QWORD *)(v7 + 16) - (_QWORD)v9 <= 3u )
  {
    v7 = sub_16E7EE0(v7, ".loh", 4u);
    v10 = *(_BYTE **)(v7 + 24);
    if ( *(_BYTE **)(v7 + 16) != v10 )
      goto LABEL_7;
  }
  else
  {
    *v9 = 1752132654;
    v10 = (_BYTE *)(*(_QWORD *)(v7 + 24) + 4LL);
    *(_QWORD *)(v7 + 24) = v10;
    if ( *(_BYTE **)(v7 + 16) != v10 )
    {
LABEL_7:
      *v10 = 32;
      v11 = (_BYTE *)(*(_QWORD *)(v7 + 24) + 1LL);
      *(_QWORD *)(v7 + 24) = v11;
      goto LABEL_8;
    }
  }
  v7 = sub_16E7EE0(v7, " ", 1u);
  v11 = *(_BYTE **)(v7 + 24);
LABEL_8:
  v12 = *(_BYTE **)(v7 + 16);
  if ( v12 - v11 < v5 )
  {
    v26 = sub_16E7EE0(v7, v6, v5);
    v27 = *(_BYTE **)(v26 + 16);
    v7 = v26;
    v11 = *(_BYTE **)(v26 + 24);
    if ( v27 != v11 )
    {
LABEL_11:
      *v11 = 9;
      ++*(_QWORD *)(v7 + 24);
      goto LABEL_12;
    }
  }
  else
  {
    if ( v5 )
    {
      if ( (unsigned int)v5 >= 8 )
      {
        v29 = (unsigned __int64)(v11 + 8) & 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)v11 = *(_QWORD *)v6;
        *(_QWORD *)&v11[v5 - 8] = *(_QWORD *)&v6[v5 - 8];
        v30 = &v11[-v29];
        v31 = (char *)(v6 - v30);
        if ( (((_DWORD)v5 + (_DWORD)v30) & 0xFFFFFFF8) >= 8 )
        {
          v32 = (v5 + (_DWORD)v30) & 0xFFFFFFF8;
          v33 = 0;
          do
          {
            v34 = v33;
            v33 += 8;
            *(_QWORD *)(v29 + v34) = *(_QWORD *)&v31[v34];
          }
          while ( v33 < v32 );
        }
        v11 = (_BYTE *)(v5 + *(_QWORD *)(v7 + 24));
        v12 = *(_BYTE **)(v7 + 16);
        *(_QWORD *)(v7 + 24) = v11;
      }
      else
      {
        *(_DWORD *)v11 = *(_DWORD *)v6;
        *(_DWORD *)&v11[(unsigned int)v5 - 4] = *(_DWORD *)&v6[(unsigned int)v5 - 4];
        v12 = *(_BYTE **)(v7 + 16);
        v11 = (_BYTE *)(v5 + *(_QWORD *)(v7 + 24));
        *(_QWORD *)(v7 + 24) = v11;
      }
    }
    if ( v12 != v11 )
      goto LABEL_11;
  }
  sub_16E7EE0(v7, "\t", 1u);
LABEL_12:
  v13 = *(_BYTE ***)a3;
  v14 = *(_QWORD *)a3 + 8LL * *(unsigned int *)(a3 + 8);
  if ( (_BYTE **)v14 != v13 )
  {
    v15 = *v13;
    for ( i = v13 + 1; ; ++i )
    {
      sub_38E2490(v15, *(_QWORD *)(a1 + 272), *(_BYTE **)(a1 + 280));
      if ( (_BYTE **)v14 == i )
        break;
      v17 = *(_QWORD *)(a1 + 272);
      v15 = *i;
      v18 = *(_WORD **)(v17 + 24);
      if ( *(_QWORD *)(v17 + 16) - (_QWORD)v18 <= 1u )
      {
        sub_16E7EE0(v17, ", ", 2u);
      }
      else
      {
        *v18 = 8236;
        *(_QWORD *)(v17 + 24) += 2LL;
      }
    }
  }
  v19 = *(unsigned int *)(a1 + 312);
  if ( *(_DWORD *)(a1 + 312) )
  {
    v22 = *(_QWORD *)(a1 + 272);
    v23 = *(char **)(a1 + 304);
    v24 = *(unsigned int *)(a1 + 312);
    v25 = *(void **)(v22 + 24);
    if ( v19 > *(_QWORD *)(v22 + 16) - (_QWORD)v25 )
    {
      sub_16E7EE0(*(_QWORD *)(a1 + 272), v23, v24);
    }
    else
    {
      memcpy(v25, v23, v24);
      *(_QWORD *)(v22 + 24) += v19;
    }
  }
  *(_DWORD *)(a1 + 312) = 0;
  if ( (*(_BYTE *)(a1 + 680) & 1) != 0 )
    return sub_39E0440(a1);
  v21 = *(_QWORD *)(a1 + 272);
  result = *(_BYTE **)(v21 + 24);
  if ( (unsigned __int64)result >= *(_QWORD *)(v21 + 16) )
    return (_BYTE *)sub_16E7DE0(v21, 10);
  *(_QWORD *)(v21 + 24) = result + 1;
  *result = 10;
  return result;
}
