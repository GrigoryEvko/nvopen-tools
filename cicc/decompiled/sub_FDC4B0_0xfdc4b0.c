// Function: sub_FDC4B0
// Address: 0xfdc4b0
//
__int64 __fastcall sub_FDC4B0(__int64 a1)
{
  __int64 result; // rax

  result = 0;
  if ( *(_QWORD *)a1 )
    return *(_QWORD *)(*(_QWORD *)(*(_QWORD *)a1 + 8LL) + 16LL);
  return result;
}
