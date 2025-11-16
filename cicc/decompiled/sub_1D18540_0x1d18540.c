// Function: sub_1D18540
// Address: 0x1d18540
//
__int64 __fastcall sub_1D18540(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 v6; // rbx
  __int64 result; // rax

  v6 = *(_QWORD *)(a1 + 648);
  result = *(unsigned int *)(v6 + 656);
  if ( (unsigned int)result >= *(_DWORD *)(v6 + 660) )
  {
    sub_16CD150(v6 + 648, (const void *)(v6 + 664), 0, 8, a5, a6);
    result = *(unsigned int *)(v6 + 656);
  }
  *(_QWORD *)(*(_QWORD *)(v6 + 648) + 8 * result) = a2;
  ++*(_DWORD *)(v6 + 656);
  return result;
}
