// Function: sub_7DFA70
// Address: 0x7dfa70
//
__int64 __fastcall sub_7DFA70(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 i; // rax
  _QWORD *v7; // r14
  _BOOL4 v8; // r13d
  __int64 result; // rax
  __int64 v10; // rbx
  __int64 j; // r15
  __int64 v12; // rdi
  __int64 v13; // rsi

  for ( i = a1[16]; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v7 = *(_QWORD **)(i + 168);
  v8 = 0;
  result = v7[3];
  if ( result )
    v8 = *v7 != result;
  v10 = a1[22];
  for ( j = 0; v10; v10 = *(_QWORD *)(v10 + 120) )
  {
    if ( *(_BYTE *)(v10 + 173) == 10 )
    {
      result = sub_8D3AA0(*(_QWORD *)(v10 + 128));
      if ( (_DWORD)result )
        result = sub_7DFA70(v10);
    }
    if ( v8 )
    {
      if ( *(char *)(v10 + 171) < 0
        && ((result = v7[3], v12 = *(_QWORD *)(v10 + 128), v13 = *(_QWORD *)(result + 40), v12 == v13)
         || (result = sub_8D97D0(v12, v13, 0, a4, a5), (_DWORD)result)) )
      {
        if ( a1[22] != v10
          && (*(_QWORD *)(j + 120) = *(_QWORD *)(v10 + 120),
              result = a1[22],
              *(_QWORD *)(v10 + 120) = result,
              a1[22] = v10,
              a1[23] == v10) )
        {
          a1[23] = j;
          v8 = 0;
          j = v10;
        }
        else
        {
          j = v10;
          v8 = 0;
        }
      }
      else
      {
        j = v10;
      }
    }
  }
  return result;
}
