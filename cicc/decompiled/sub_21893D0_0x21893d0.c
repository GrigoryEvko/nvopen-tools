// Function: sub_21893D0
// Address: 0x21893d0
//
const char *__fastcall sub_21893D0(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v4; // rdx
  __int64 v5; // rdx
  const char *result; // rax
  __int64 v7; // rdx
  char *v8; // r12
  size_t v9; // rax
  void *v10; // rdi
  size_t v11; // r14
  _WORD *v12; // rdx
  __int64 v13; // rdx
  _WORD *v14; // rdx
  __int64 v15; // rdx
  _WORD *v16; // rdx
  __int64 v17; // rdx
  _WORD *v18; // rdx

  v4 = a3 >> 28;
  if ( a3 > 0x9FFFFFFF )
    sub_16BD130("Bad virtual register encoding", 1u);
  switch ( v4 )
  {
    case 0LL:
      result = sub_2189340(a3);
      v8 = (char *)result;
      if ( result )
      {
        v9 = strlen(result);
        v10 = *(void **)(a2 + 24);
        v11 = v9;
        result = (const char *)(*(_QWORD *)(a2 + 16) - (_QWORD)v10);
        if ( v11 > (unsigned __int64)result )
        {
          return (const char *)sub_16E7EE0(a2, v8, v11);
        }
        else if ( v11 )
        {
          result = (const char *)memcpy(v10, v8, v11);
          *(_QWORD *)(a2 + 24) += v11;
        }
      }
      return result;
    case 1LL:
      v12 = *(_WORD **)(a2 + 24);
      if ( *(_QWORD *)(a2 + 16) - (_QWORD)v12 <= 1u )
      {
        sub_16E7EE0(a2, "%p", 2u);
      }
      else
      {
        *v12 = 28709;
        *(_QWORD *)(a2 + 24) += 2LL;
      }
      return (const char *)sub_16E7A90(a2, a3 & 0xFFFFFFF);
    case 2LL:
      v13 = *(_QWORD *)(a2 + 24);
      if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v13) <= 2 )
      {
        sub_16E7EE0(a2, "%rs", 3u);
      }
      else
      {
        *(_BYTE *)(v13 + 2) = 115;
        *(_WORD *)v13 = 29221;
        *(_QWORD *)(a2 + 24) += 3LL;
      }
      return (const char *)sub_16E7A90(a2, a3 & 0xFFFFFFF);
    case 3LL:
      v14 = *(_WORD **)(a2 + 24);
      if ( *(_QWORD *)(a2 + 16) - (_QWORD)v14 <= 1u )
      {
        sub_16E7EE0(a2, "%r", 2u);
      }
      else
      {
        *v14 = 29221;
        *(_QWORD *)(a2 + 24) += 2LL;
      }
      return (const char *)sub_16E7A90(a2, a3 & 0xFFFFFFF);
    case 4LL:
      v15 = *(_QWORD *)(a2 + 24);
      if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v15) <= 2 )
      {
        sub_16E7EE0(a2, "%rd", 3u);
      }
      else
      {
        *(_BYTE *)(v15 + 2) = 100;
        *(_WORD *)v15 = 29221;
        *(_QWORD *)(a2 + 24) += 3LL;
      }
      return (const char *)sub_16E7A90(a2, a3 & 0xFFFFFFF);
    case 5LL:
      v16 = *(_WORD **)(a2 + 24);
      if ( *(_QWORD *)(a2 + 16) - (_QWORD)v16 <= 1u )
      {
        sub_16E7EE0(a2, "%f", 2u);
      }
      else
      {
        *v16 = 26149;
        *(_QWORD *)(a2 + 24) += 2LL;
      }
      return (const char *)sub_16E7A90(a2, a3 & 0xFFFFFFF);
    case 6LL:
      v17 = *(_QWORD *)(a2 + 24);
      if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v17) <= 2 )
      {
        sub_16E7EE0(a2, "%fd", 3u);
      }
      else
      {
        *(_BYTE *)(v17 + 2) = 100;
        *(_WORD *)v17 = 26149;
        *(_QWORD *)(a2 + 24) += 3LL;
      }
      return (const char *)sub_16E7A90(a2, a3 & 0xFFFFFFF);
    case 7LL:
      v18 = *(_WORD **)(a2 + 24);
      if ( *(_QWORD *)(a2 + 16) - (_QWORD)v18 <= 1u )
      {
        sub_16E7EE0(a2, "%h", 2u);
      }
      else
      {
        *v18 = 26661;
        *(_QWORD *)(a2 + 24) += 2LL;
      }
      return (const char *)sub_16E7A90(a2, a3 & 0xFFFFFFF);
    case 8LL:
      v5 = *(_QWORD *)(a2 + 24);
      if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v5) <= 2 )
      {
        sub_16E7EE0(a2, "%hh", 3u);
      }
      else
      {
        *(_BYTE *)(v5 + 2) = 104;
        *(_WORD *)v5 = 26661;
        *(_QWORD *)(a2 + 24) += 3LL;
      }
      return (const char *)sub_16E7A90(a2, a3 & 0xFFFFFFF);
    case 9LL:
      v7 = *(_QWORD *)(a2 + 24);
      if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v7) <= 2 )
      {
        sub_16E7EE0(a2, "%rq", 3u);
      }
      else
      {
        *(_BYTE *)(v7 + 2) = 113;
        *(_WORD *)v7 = 29221;
        *(_QWORD *)(a2 + 24) += 3LL;
      }
      return (const char *)sub_16E7A90(a2, a3 & 0xFFFFFFF);
  }
}
