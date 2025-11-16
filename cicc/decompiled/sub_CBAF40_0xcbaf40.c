// Function: sub_CBAF40
// Address: 0xcbaf40
//
_BYTE *__fastcall sub_CBAF40(__int64 a1, _BYTE *a2, unsigned __int64 a3, __int64 a4, __int64 a5)
{
  _BYTE *result; // rax
  __int64 v7; // rdi
  __int64 v8; // rdx
  __int64 v9; // rcx

  result = (_BYTE *)(a3 - 1);
  if ( a4 < a5 )
  {
    v7 = *(_QWORD *)(a1 + 8);
    while ( 1 )
    {
      v8 = *(_QWORD *)(v7 + 8 * a4);
      if ( (v8 & 0xF8000000) != 0x70000000 )
        break;
      if ( a5 == ++a4 )
        return result;
    }
    if ( (v8 & 0xF8000000) == 0x10000000 && result != a2 )
    {
      v9 = a4 + 1;
      do
      {
        if ( *result == (_BYTE)v8
          && (a5 <= v9
           || a3 <= (unsigned __int64)(result + 1)
           || (*(_QWORD *)(v7 + 8 * v9) & 0xF8000000LL) != 0x10000000
           || result[1] == (unsigned __int8)*(_QWORD *)(v7 + 8 * v9)) )
        {
          break;
        }
        --result;
      }
      while ( a2 != result );
    }
  }
  return result;
}
