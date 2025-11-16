// Function: sub_2E3A060
// Address: 0x2e3a060
//
__int64 __fastcall sub_2E3A060(__int64 *a1)
{
  __int64 result; // rax

  result = *a1;
  if ( *a1 )
    return *(_QWORD *)(result + 128);
  return result;
}
