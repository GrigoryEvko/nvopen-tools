// Function: sub_21E69B0
// Address: 0x21e69b0
//
_QWORD *__fastcall sub_21E69B0(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, const char *a5)
{
  _QWORD *result; // rax
  __int64 v8; // rcx
  unsigned __int64 v9; // rdx
  _WORD *v10; // rdx
  char *v11; // rsi

  result = (_QWORD *)(*(_QWORD *)(a2 + 16) + 16LL * a3);
  if ( a5 )
  {
    if ( !strcmp(a5, "bypass") )
    {
      v8 = *(_QWORD *)(a4 + 24);
      v9 = *(_QWORD *)(a4 + 16) - v8;
      if ( result[1] )
      {
        if ( v9 > 2 )
        {
          *(_BYTE *)(v8 + 2) = 103;
          *(_WORD *)v8 = 25390;
          *(_QWORD *)(a4 + 24) += 3LL;
          return result;
        }
        v11 = (char *)&unk_435F090;
      }
      else
      {
        if ( v9 > 2 )
        {
          *(_BYTE *)(v8 + 2) = 97;
          *(_WORD *)v8 = 25390;
          *(_QWORD *)(a4 + 24) += 3LL;
          return result;
        }
        v11 = (char *)&unk_435F08C;
      }
      return (_QWORD *)sub_16E7EE0(a4, v11, 3u);
    }
    if ( !strcmp(a5, "srcsize") && (*(_BYTE *)result != 2 || result[1] != -1) )
    {
      v10 = *(_WORD **)(a4 + 24);
      if ( *(_QWORD *)(a4 + 16) - (_QWORD)v10 <= 1u )
      {
        sub_16E7EE0(a4, ", ", 2u);
      }
      else
      {
        *v10 = 8236;
        *(_QWORD *)(a4 + 24) += 2LL;
      }
      return sub_21897A0(a1, a2, a3, a4);
    }
  }
  return result;
}
