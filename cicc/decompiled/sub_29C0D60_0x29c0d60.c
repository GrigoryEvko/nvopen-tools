// Function: sub_29C0D60
// Address: 0x29c0d60
//
unsigned __int64 __fastcall sub_29C0D60(__int64 a1)
{
  unsigned __int64 result; // rax
  unsigned __int64 v2; // rdx

  result = sub_AA4E50(a1);
  if ( !result )
  {
    result = sub_AA4F10(a1);
    if ( !result )
    {
      v2 = *(_QWORD *)(a1 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v2 != a1 + 48 )
      {
        if ( !v2 )
          BUG();
        if ( (unsigned int)*(unsigned __int8 *)(v2 - 24) - 30 <= 0xA )
          return v2 - 24;
      }
    }
  }
  return result;
}
