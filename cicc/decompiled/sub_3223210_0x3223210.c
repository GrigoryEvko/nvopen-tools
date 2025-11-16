// Function: sub_3223210
// Address: 0x3223210
//
__int64 __fastcall sub_3223210(__int64 a1)
{
  int v1; // edx
  __int64 result; // rax

  v1 = *(_DWORD *)(a1 + 3760);
  result = 1;
  if ( v1 != 2 )
  {
    result = 0;
    if ( !v1 )
      return *(unsigned __int8 *)(a1 + 3769);
  }
  return result;
}
