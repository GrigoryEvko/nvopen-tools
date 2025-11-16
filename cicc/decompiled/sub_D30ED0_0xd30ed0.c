// Function: sub_D30ED0
// Address: 0xd30ed0
//
__int64 __fastcall sub_D30ED0(__int64 a1)
{
  unsigned __int16 v1; // ax
  __int64 result; // rax

  v1 = *(_WORD *)(a1 + 2);
  if ( ((v1 >> 7) & 6) != 0 )
    return 1;
  result = v1 & 1;
  if ( !(_DWORD)result )
    return sub_D2F6D0(a1);
  return result;
}
