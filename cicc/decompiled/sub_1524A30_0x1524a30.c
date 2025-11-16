// Function: sub_1524A30
// Address: 0x1524a30
//
__int64 __fastcall sub_1524A30(__int64 a1, __int64 a2, int a3, __int64 a4)
{
  int v5; // r12d
  __int64 result; // rax

  v5 = a3 - sub_153E840(a1 + 24);
  result = *(unsigned int *)(a4 + 8);
  if ( (unsigned int)result >= *(_DWORD *)(a4 + 12) )
  {
    sub_16CD150(a4, a4 + 16, 0, 4);
    result = *(unsigned int *)(a4 + 8);
  }
  *(_DWORD *)(*(_QWORD *)a4 + 4 * result) = v5;
  ++*(_DWORD *)(a4 + 8);
  return result;
}
