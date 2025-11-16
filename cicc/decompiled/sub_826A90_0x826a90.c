// Function: sub_826A90
// Address: 0x826a90
//
__int64 __fastcall sub_826A90(_DWORD *a1, __int64 a2)
{
  __int64 result; // rax

  result = 0x100000000002000LL;
  if ( (*(_QWORD *)(a2 + 192) & 0x100000000002000LL) == 0x100000000002000LL && *(_BYTE *)(a2 + 174) == 5 )
  {
    result = (unsigned int)*(unsigned __int8 *)(a2 + 176) - 1;
    if ( (unsigned __int8)(*(_BYTE *)(a2 + 176) - 1) <= 3u )
      return sub_684AA0(4u, 0xD9Eu, a1);
  }
  return result;
}
