// Function: sub_1AE75D0
// Address: 0x1ae75d0
//
__int64 __fastcall sub_1AE75D0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rdx

  v4 = *(_QWORD *)(*(_QWORD *)(a3 + 24 * (2LL - (*(_DWORD *)(a3 + 20) & 0xFFFFFFF))) + 24LL);
  *(_BYTE *)(a1 + 8) = 1;
  *(_QWORD *)a1 = v4;
  return a1;
}
