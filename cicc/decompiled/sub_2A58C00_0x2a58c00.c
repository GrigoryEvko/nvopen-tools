// Function: sub_2A58C00
// Address: 0x2a58c00
//
__int64 __fastcall sub_2A58C00(_QWORD *a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rbx
  __int64 result; // rax

  v7 = *a1 + 104LL * a2;
  result = *(unsigned int *)(v7 + 40);
  if ( result + 1 > (unsigned __int64)*(unsigned int *)(v7 + 44) )
  {
    sub_C8D5F0(v7 + 32, (const void *)(v7 + 48), result + 1, 8u, a5, a6);
    result = *(unsigned int *)(v7 + 40);
  }
  *(_QWORD *)(*(_QWORD *)(v7 + 32) + 8 * result) = a3;
  ++*(_DWORD *)(v7 + 40);
  return result;
}
