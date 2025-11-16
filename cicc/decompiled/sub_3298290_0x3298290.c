// Function: sub_3298290
// Address: 0x3298290
//
__int64 __fastcall sub_3298290(__int64 a1, __int64 *a2)
{
  __int64 v3; // rdx

  *(_DWORD *)(a1 + 16) = 64;
  *(_QWORD *)(a1 + 8) = 0;
  v3 = *a2;
  *(_DWORD *)a1 = 57;
  *(_QWORD *)(a1 + 24) = v3;
  *(_BYTE *)(a1 + 36) = 0;
  return a1;
}
