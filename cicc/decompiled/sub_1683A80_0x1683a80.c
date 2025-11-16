// Function: sub_1683A80
// Address: 0x1683a80
//
__int64 __fastcall sub_1683A80(__int64 a1)
{
  __int64 v1; // r12

  v1 = *(_QWORD *)(a1 + 8);
  *(_DWORD *)(v1 + 4) = sub_16837A0(v1);
  return (unsigned int)_InterlockedCompareExchange(*(volatile signed __int32 **)(a1 + 8), 11, 0);
}
