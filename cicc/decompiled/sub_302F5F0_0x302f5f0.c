// Function: sub_302F5F0
// Address: 0x302f5f0
//
__int64 __fastcall sub_302F5F0(__int64 a1, char *a2, __int64 a3)
{
  char v3; // al
  bool v4; // zf
  __int64 result; // rax

  if ( a3 != 1 )
    return 0;
  v3 = *a2;
  if ( *a2 == 109 )
    return 4;
  if ( v3 == 111 )
    return 5;
  if ( v3 == 88 )
    return 19;
  v4 = v3 == 112;
  result = 0;
  if ( v4 )
    return 24;
  return result;
}
