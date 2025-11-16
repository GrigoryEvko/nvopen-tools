// Function: sub_E4E2F0
// Address: 0xe4e2f0
//
_BYTE *__fastcall sub_E4E2F0(__int64 a1, int a2, __int64 a3)
{
  size_t v5; // r14
  char *v6; // r15
  __int64 v7; // rdi
  _BYTE *v8; // rax
  _DWORD *v9; // rdx
  _BYTE *v10; // rax
  char *v11; // rax
  char *v12; // rdx
  __int64 *v13; // rbx
  __int64 v14; // r13
  __int64 v15; // r15
  __int64 *i; // rbx
  __int64 v17; // rdi
  _WORD *v18; // rdx
  __int64 v20; // rax
  __int64 v21; // rax
  unsigned __int64 v22; // rsi
  char *v23; // rax
  char *v24; // r15
  unsigned int v25; // edx
  unsigned int v26; // eax
  __int64 v27; // rcx

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
  v7 = *(_QWORD *)(a1 + 304);
  v8 = *(_BYTE **)(v7 + 32);
  if ( *(_BYTE **)(v7 + 24) == v8 )
  {
    v21 = sub_CB6200(v7, (unsigned __int8 *)"\t", 1u);
    v9 = *(_DWORD **)(v21 + 32);
    v7 = v21;
  }
  else
  {
    *v8 = 9;
    v9 = (_DWORD *)(*(_QWORD *)(v7 + 32) + 1LL);
    *(_QWORD *)(v7 + 32) = v9;
  }
  if ( *(_QWORD *)(v7 + 24) - (_QWORD)v9 <= 3u )
  {
    v7 = sub_CB6200(v7, ".loh", 4u);
    v10 = *(_BYTE **)(v7 + 32);
  }
  else
  {
    *v9 = 1752132654;
    v10 = (_BYTE *)(*(_QWORD *)(v7 + 32) + 4LL);
    *(_QWORD *)(v7 + 32) = v10;
  }
  if ( v10 == *(_BYTE **)(v7 + 24) )
  {
    v7 = sub_CB6200(v7, (unsigned __int8 *)" ", 1u);
    v11 = *(char **)(v7 + 32);
  }
  else
  {
    *v10 = 32;
    v11 = (char *)(*(_QWORD *)(v7 + 32) + 1LL);
    *(_QWORD *)(v7 + 32) = v11;
  }
  v12 = *(char **)(v7 + 24);
  if ( v12 - v11 < v5 )
  {
    v20 = sub_CB6200(v7, (unsigned __int8 *)v6, v5);
    v12 = *(char **)(v20 + 24);
    v7 = v20;
    v11 = *(char **)(v20 + 32);
  }
  else if ( v5 )
  {
    if ( (unsigned int)v5 >= 8 )
    {
      v22 = (unsigned __int64)(v11 + 8) & 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)v11 = *(_QWORD *)v6;
      *(_QWORD *)&v11[v5 - 8] = *(_QWORD *)&v6[v5 - 8];
      v23 = &v11[-v22];
      v24 = (char *)(v6 - v23);
      if ( (((_DWORD)v5 + (_DWORD)v23) & 0xFFFFFFF8) >= 8 )
      {
        v25 = (v5 + (_DWORD)v23) & 0xFFFFFFF8;
        v26 = 0;
        do
        {
          v27 = v26;
          v26 += 8;
          *(_QWORD *)(v22 + v27) = *(_QWORD *)&v24[v27];
        }
        while ( v26 < v25 );
      }
      v11 = (char *)(v5 + *(_QWORD *)(v7 + 32));
      v12 = *(char **)(v7 + 24);
      *(_QWORD *)(v7 + 32) = v11;
    }
    else
    {
      *(_DWORD *)v11 = *(_DWORD *)v6;
      *(_DWORD *)&v11[(unsigned int)v5 - 4] = *(_DWORD *)&v6[(unsigned int)v5 - 4];
      v12 = *(char **)(v7 + 24);
      v11 = (char *)(v5 + *(_QWORD *)(v7 + 32));
      *(_QWORD *)(v7 + 32) = v11;
    }
  }
  if ( v11 == v12 )
  {
    sub_CB6200(v7, (unsigned __int8 *)"\t", 1u);
  }
  else
  {
    *v11 = 9;
    ++*(_QWORD *)(v7 + 32);
  }
  v13 = *(__int64 **)a3;
  v14 = *(_QWORD *)a3 + 8LL * *(unsigned int *)(a3 + 8);
  if ( (__int64 *)v14 != v13 )
  {
    v15 = *v13;
    for ( i = v13 + 1; ; ++i )
    {
      sub_EA12C0(v15, *(_QWORD *)(a1 + 304), *(_QWORD *)(a1 + 312));
      if ( (__int64 *)v14 == i )
        break;
      v17 = *(_QWORD *)(a1 + 304);
      v15 = *i;
      v18 = *(_WORD **)(v17 + 32);
      if ( *(_QWORD *)(v17 + 24) - (_QWORD)v18 <= 1u )
      {
        sub_CB6200(v17, (unsigned __int8 *)", ", 2u);
      }
      else
      {
        *v18 = 8236;
        *(_QWORD *)(v17 + 32) += 2LL;
      }
    }
  }
  return sub_E4D880(a1);
}
