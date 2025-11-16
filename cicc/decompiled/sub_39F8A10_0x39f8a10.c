// Function: sub_39F8A10
// Address: 0x39f8a10
//
__int64 __fastcall sub_39F8A10(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v3; // rcx
  __int64 result; // rax

  v3 = *(_QWORD *)(a3 + 8);
  result = 1;
  if ( *(_QWORD *)(a2 + 8) <= v3 )
    return (unsigned int)-(*(_QWORD *)(a2 + 8) < v3);
  return result;
}
