// Function: sub_C80DE0
// Address: 0xc80de0
//
unsigned __int64 __fastcall sub_C80DE0(__int64 a1, unsigned int a2)
{
  unsigned __int64 result; // rax

  result = sub_C80CF0(*(char **)a1, *(_QWORD *)(a1 + 8), a2);
  if ( result != -1 )
    *(_QWORD *)(a1 + 8) = result;
  return result;
}
