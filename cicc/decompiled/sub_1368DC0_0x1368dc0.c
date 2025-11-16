// Function: sub_1368DC0
// Address: 0x1368dc0
//
__int64 __fastcall sub_1368DC0(__int64 a1)
{
  __int64 result; // rax

  result = 0;
  if ( *(_QWORD *)a1 )
    return *(_QWORD *)(*(_QWORD *)(*(_QWORD *)a1 + 8LL) + 16LL);
  return result;
}
