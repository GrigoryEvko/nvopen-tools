// Function: sub_984300
// Address: 0x984300
//
__int64 __fastcall sub_984300(__int64 a1, __int64 *a2)
{
  __int64 result; // rax

  if ( *(_DWORD *)(a1 + 8) > 0x40u )
    return sub_C43BD0(a1, a2);
  result = *a2;
  *(_QWORD *)a1 |= *a2;
  return result;
}
