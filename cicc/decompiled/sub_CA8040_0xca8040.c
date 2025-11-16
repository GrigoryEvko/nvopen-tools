// Function: sub_CA8040
// Address: 0xca8040
//
unsigned __int64 __fastcall sub_CA8040(__int64 a1, _BYTE *a2, __int64 a3)
{
  _BYTE *v3; // r13
  _BYTE *v4; // rbx
  unsigned __int64 result; // rax

  v3 = &a2[a3];
  if ( a2 == &a2[a3] )
    return 1;
  v4 = a2;
  while ( 1 )
  {
    result = sub_CA7F80(a1, v4);
    if ( !(_BYTE)result )
      break;
    if ( ++v4 == v3 )
      return 1;
  }
  return result;
}
