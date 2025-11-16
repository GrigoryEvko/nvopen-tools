// Function: sub_155EE10
// Address: 0x155ee10
//
__int64 __fastcall sub_155EE10(__int64 a1, char a2)
{
  __int64 result; // rax

  result = 0;
  if ( *(_QWORD *)a1 )
    return (*(_QWORD *)(*(_QWORD *)a1 + 8LL) >> a2) & 1LL;
  return result;
}
