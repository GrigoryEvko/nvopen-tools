// Function: sub_1DD77B0
// Address: 0x1dd77b0
//
__int64 __fastcall sub_1DD77B0(__int64 a1)
{
  __int64 result; // rax

  result = *(_QWORD *)(a1 + 152);
  if ( result != *(_QWORD *)(a1 + 160) )
    *(_QWORD *)(a1 + 160) = result;
  return result;
}
