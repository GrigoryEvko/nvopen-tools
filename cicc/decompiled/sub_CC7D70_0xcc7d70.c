// Function: sub_CC7D70
// Address: 0xcc7d70
//
unsigned __int64 __fastcall sub_CC7D70(__int64 a1)
{
  unsigned __int64 result; // rax

  if ( *(_DWORD *)(a1 + 44) != 30 )
    BUG();
  result = sub_CC78E0(a1);
  if ( !(_DWORD)result )
    return result & 0x7FFFFFFF00000000LL | 0x8000000000000013LL;
  return result;
}
