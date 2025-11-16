// Function: sub_2E8E690
// Address: 0x2e8e690
//
__int64 __fastcall sub_2E8E690(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rdx

  result = *(unsigned int *)(a1 + 64);
  if ( !(_DWORD)result )
  {
    v2 = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 32LL);
    result = (unsigned int)(*(_DWORD *)(v2 + 896) + 1);
    *(_DWORD *)(v2 + 896) = result;
    *(_DWORD *)(a1 + 64) = result;
  }
  return result;
}
