// Function: sub_329E1B0
// Address: 0x329e1b0
//
__int64 __fastcall sub_329E1B0(__int64 a1, __int64 a2)
{
  unsigned int v2; // eax
  __int64 result; // rax

  *(_QWORD *)a1 = *(_QWORD *)a2;
  *(_DWORD *)(a1 + 8) = *(_DWORD *)(a2 + 8);
  v2 = *(_DWORD *)(a2 + 24);
  *(_DWORD *)(a1 + 24) = v2;
  if ( v2 > 0x40 )
    sub_C43780(a1 + 16, (const void **)(a2 + 16));
  else
    *(_QWORD *)(a1 + 16) = *(_QWORD *)(a2 + 16);
  *(_QWORD *)(a1 + 32) = *(_QWORD *)(a2 + 32);
  result = *(_QWORD *)(a2 + 40);
  *(_QWORD *)(a1 + 40) = result;
  return result;
}
