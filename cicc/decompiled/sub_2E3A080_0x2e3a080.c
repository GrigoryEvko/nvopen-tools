// Function: sub_2E3A080
// Address: 0x2e3a080
//
__int64 __fastcall sub_2E3A080(__int64 a1)
{
  __int64 result; // rax

  result = 0;
  if ( *(_QWORD *)a1 )
    return *(_QWORD *)(*(_QWORD *)(*(_QWORD *)a1 + 8LL) + 16LL);
  return result;
}
