// Function: sub_2F290F0
// Address: 0x2f290f0
//
__int64 __fastcall sub_2F290F0(__int64 a1, int a2, __int16 a3)
{
  int v3; // ebx
  unsigned int *v4; // r12

  v3 = a3 & 0xFFF;
  v4 = (unsigned int *)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 32LL) + 40LL * *(unsigned int *)(a1 + 16));
  sub_2EAB0C0((__int64)v4, a2);
  *v4 = (v3 << 8) | *v4 & 0xFFF000FF;
  return 1;
}
