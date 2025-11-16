// Function: sub_1F40590
// Address: 0x1f40590
//
__int64 __fastcall sub_1F40590(__int64 a1)
{
  unsigned __int64 v1; // rdi
  __int64 result; // rax

  v1 = a1 - 1;
  result = 462;
  if ( v1 <= 0xF )
    return dword_42F2F00[v1];
  return result;
}
