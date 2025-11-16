// Function: sub_386F050
// Address: 0x386f050
//
__int64 __fastcall sub_386F050(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  unsigned __int8 v3; // dl

  result = *(_QWORD *)(a1 + 32);
  if ( *(_BYTE *)(a1 + 16) == 29 )
    result = *(_QWORD *)(*(_QWORD *)(a1 - 48) + 48LL);
  while ( 1 )
  {
    if ( !result )
      BUG();
    v3 = *(_BYTE *)(result - 8);
    if ( v3 != 77 )
      break;
    result = *(_QWORD *)(result + 8);
  }
  if ( (unsigned int)v3 - 73 <= 1 || v3 == 88 )
    return *(_QWORD *)(result + 8);
  if ( v3 == 34 )
    return sub_157EE30(a2);
  return result;
}
