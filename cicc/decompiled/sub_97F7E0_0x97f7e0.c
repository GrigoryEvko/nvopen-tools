// Function: sub_97F7E0
// Address: 0x97f7e0
//
__int64 __fastcall sub_97F7E0(_QWORD *a1)
{
  int v1; // ecx
  unsigned __int64 v2; // rdi

  *a1 = 0;
  v1 = (int)a1;
  v2 = (unsigned __int64)(a1 + 1);
  *(_QWORD *)(v2 + 115) = 0;
  memset((void *)(v2 & 0xFFFFFFFFFFFFFFF8LL), 0, 8 * ((v1 - (v2 & 0xFFFFFFF8) + 131) >> 3));
  return 0;
}
