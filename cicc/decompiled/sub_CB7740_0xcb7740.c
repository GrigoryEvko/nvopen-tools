// Function: sub_CB7740
// Address: 0xcb7740
//
unsigned __int8 *__fastcall sub_CB7740(__int64 a1, unsigned __int64 a2)
{
  unsigned __int8 *result; // rax

  if ( a2 <= 0x1FFFFFFFFFFFFFFFLL && (result = (unsigned __int8 *)realloc(*(void **)(a1 + 24))) != 0 )
  {
    *(_QWORD *)(a1 + 32) = a2;
    *(_QWORD *)(a1 + 24) = result;
  }
  else
  {
    if ( !*(_DWORD *)(a1 + 16) )
      *(_DWORD *)(a1 + 16) = 12;
    *(_QWORD *)a1 = byte_4F85140;
    *(_QWORD *)(a1 + 8) = byte_4F85140;
    return byte_4F85140;
  }
  return result;
}
