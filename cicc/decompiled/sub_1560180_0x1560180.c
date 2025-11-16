// Function: sub_1560180
// Address: 0x1560180
//
__int64 __fastcall sub_1560180(__int64 a1, char a2)
{
  __int64 result; // rax

  result = 0;
  if ( *(_QWORD *)a1 )
    return (*(_QWORD *)(*(_QWORD *)a1 + 8LL) >> a2) & 1LL;
  return result;
}
