// Function: sub_AA5190
// Address: 0xaa5190
//
__int64 __fastcall sub_AA5190(__int64 a1)
{
  __int64 result; // rax
  unsigned __int64 v2; // rcx
  __int64 v3; // rsi

  result = sub_AA4FF0(a1);
  if ( result != a1 + 48 )
  {
    if ( !result )
      BUG();
    v2 = (unsigned int)*(unsigned __int8 *)(result - 24) - 39;
    if ( (unsigned int)v2 <= 0x38 )
    {
      v3 = 0x100060000000001LL;
      if ( _bittest64(&v3, v2) )
        return *(_QWORD *)(result + 8);
    }
  }
  return result;
}
