// Function: sub_264DDF0
// Address: 0x264ddf0
//
_DWORD *__fastcall sub_264DDF0(__int64 a1)
{
  _DWORD *result; // rax
  _DWORD *i; // rdx

  result = *(_DWORD **)(a1 + 16);
  for ( i = *(_DWORD **)(a1 + 24); result != i; *(_QWORD *)(a1 + 16) = result )
  {
    if ( *result <= 0xFFFFFFFD )
      break;
    ++result;
  }
  return result;
}
