// Function: sub_FDC440
// Address: 0xfdc440
//
__int64 __fastcall sub_FDC440(__int64 *a1)
{
  __int64 result; // rax

  result = *a1;
  if ( *a1 )
    return *(_QWORD *)(result + 128);
  return result;
}
