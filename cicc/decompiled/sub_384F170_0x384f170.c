// Function: sub_384F170
// Address: 0x384f170
//
__int64 __fastcall sub_384F170(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  *(_DWORD *)(a1 + 76) += *(_DWORD *)(a2 + 8);
  *(_DWORD *)(a1 + 556) -= *(_DWORD *)(a2 + 8);
  result = *(unsigned int *)(a2 + 8);
  *(_DWORD *)(a1 + 560) += result;
  *(_QWORD *)a2 = -16;
  --*(_DWORD *)(a1 + 216);
  ++*(_DWORD *)(a1 + 220);
  if ( *(_BYTE *)(a1 + 352) )
  {
    result = *(unsigned int *)(a1 + 528);
    *(_DWORD *)(a1 + 76) += result;
    *(_DWORD *)(a1 + 528) = 0;
    *(_BYTE *)(a1 + 352) = 0;
  }
  return result;
}
