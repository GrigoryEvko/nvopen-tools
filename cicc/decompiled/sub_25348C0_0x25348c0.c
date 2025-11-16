// Function: sub_25348C0
// Address: 0x25348c0
//
__int64 __fastcall sub_25348C0(__int64 a1, char a2)
{
  __int64 result; // rax

  if ( !a2 )
  {
    result = *(unsigned __int8 *)(a1 + 8);
    *(_BYTE *)(a1 + 9) = result;
  }
  return result;
}
