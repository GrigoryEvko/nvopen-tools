// Function: sub_D48970
// Address: 0xd48970
//
__int64 __fastcall sub_D48970(__int64 a1)
{
  __int64 result; // rax
  unsigned __int64 v2; // rdx

  result = sub_D47930(a1);
  if ( result )
  {
    v2 = *(_QWORD *)(result + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v2 == result + 48 )
      return 0;
    if ( !v2 )
      BUG();
    if ( *(_BYTE *)(v2 - 24) != 31 )
    {
      return 0;
    }
    else
    {
      result = 0;
      if ( (*(_DWORD *)(v2 - 20) & 0x7FFFFFF) == 3 )
      {
        result = *(_QWORD *)(v2 - 120);
        if ( *(_BYTE *)result != 82 )
          return 0;
      }
    }
  }
  return result;
}
