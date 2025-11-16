// Function: sub_9C8C60
// Address: 0x9c8c60
//
__int64 __fastcall sub_9C8C60(__int64 a1, int a2)
{
  __int64 result; // rax

  result = *(unsigned int *)(a1 + 8);
  if ( result + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
  {
    sub_C8D5F0(a1, a1 + 16, result + 1, 4);
    result = *(unsigned int *)(a1 + 8);
  }
  *(_DWORD *)(*(_QWORD *)a1 + 4 * result) = a2;
  ++*(_DWORD *)(a1 + 8);
  return result;
}
