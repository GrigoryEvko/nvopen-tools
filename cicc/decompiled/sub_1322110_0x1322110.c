// Function: sub_1322110
// Address: 0x1322110
//
_QWORD *__fastcall sub_1322110(_BYTE *a1, __int64 a2, char a3, char a4)
{
  __int64 v4; // rax
  _QWORD *result; // rax
  __int64 v8; // rax
  __int64 v9; // rsi
  __int64 v10; // rcx

  v4 = 0;
  if ( a2 != 4096 )
  {
    v4 = 1;
    if ( a2 != 4097 )
    {
      if ( !a3 || (v4 = 0, a2 != *(_DWORD *)(qword_4F96BA0 + 8)) )
        v4 = (unsigned int)(a2 + 2);
    }
  }
  result = *(_QWORD **)(qword_4F96BA0 + 8 * v4 + 24);
  if ( !result && a4 )
  {
    v8 = sub_131BF10();
    result = sub_131C440(a1, v8, 37856, 16);
    if ( result )
    {
      *(_DWORD *)result = a2;
      v9 = qword_4F96BA0;
      result[10] = result + 11;
      v10 = 0;
      if ( a2 != 4096 )
      {
        v10 = 1;
        if ( a2 != 4097 )
        {
          if ( !a3 || (v10 = 0, a2 != *(_DWORD *)(v9 + 8)) )
            v10 = (unsigned int)(a2 + 2);
        }
      }
      *(_QWORD *)(v9 + 8 * v10 + 24) = result;
    }
    else
    {
      return 0;
    }
  }
  return result;
}
