// Function: sub_1E738C0
// Address: 0x1e738c0
//
__int64 __fastcall sub_1E738C0(int a1, int a2, __int64 a3, __int64 a4, unsigned __int8 a5)
{
  __int64 result; // rax

  if ( a1 < a2 )
  {
    *(_BYTE *)(a3 + 24) = a5;
    return 1;
  }
  else
  {
    result = 0;
    if ( a1 > a2 )
    {
      result = 1;
      if ( *(_BYTE *)(a4 + 24) > a5 )
        *(_BYTE *)(a4 + 24) = a5;
    }
  }
  return result;
}
