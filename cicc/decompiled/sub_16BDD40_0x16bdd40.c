// Function: sub_16BDD40
// Address: 0x16bdd40
//
unsigned __int64 __fastcall sub_16BDD40(unsigned __int64 **a1)
{
  unsigned __int64 result; // rax
  unsigned __int64 *v2; // rdx

  result = **a1;
  if ( !result || (result & 1) != 0 )
  {
    v2 = (unsigned __int64 *)((result & 0xFFFFFFFFFFFFFFFELL) + 8);
    result = *v2;
    if ( *v2 != -1 )
    {
      while ( !result || (result & 1) != 0 )
      {
        result = v2[1];
        if ( result == -1 )
          break;
        ++v2;
      }
    }
  }
  *a1 = (unsigned __int64 *)result;
  return result;
}
