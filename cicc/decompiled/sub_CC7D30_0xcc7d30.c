// Function: sub_CC7D30
// Address: 0xcc7d30
//
unsigned __int64 __fastcall sub_CC7D30(__int64 a1)
{
  int v1; // eax
  unsigned __int64 result; // rax

  v1 = *(_DWORD *)(a1 + 44);
  if ( v1 == 9 )
    return 2;
  if ( v1 != 28 )
  {
    if ( v1 != 1 )
      BUG();
    return 2;
  }
  result = sub_CC78E0(a1);
  if ( !(_DWORD)result )
    return 2;
  return result;
}
