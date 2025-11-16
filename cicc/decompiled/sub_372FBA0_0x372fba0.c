// Function: sub_372FBA0
// Address: 0x372fba0
//
__int64 __fastcall sub_372FBA0(__int64 a1)
{
  __int64 result; // rax

  result = *(_QWORD *)(a1 + 144) + 32LL * *(unsigned int *)(a1 + 152) - 32;
  if ( *(_QWORD *)(result + 16) == *(_QWORD *)(a1 + 1192) )
  {
    result = sub_372FA10(a1 + 1464, *(_QWORD *)(a1 + 1464) + 32LL * *(_QWORD *)(result + 24), *(_QWORD *)(a1 + 1472));
    --*(_DWORD *)(a1 + 152);
  }
  return result;
}
