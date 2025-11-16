// Function: sub_149CBC0
// Address: 0x149cbc0
//
__int64 __fastcall sub_149CBC0(_QWORD *a1)
{
  int v1; // ecx
  unsigned __int64 v2; // rdi

  *a1 = 0;
  v1 = (int)a1;
  v2 = (unsigned __int64)(a1 + 1);
  *(_QWORD *)(v2 + 90) = 0;
  memset((void *)(v2 & 0xFFFFFFFFFFFFFFF8LL), 0, 8 * ((v1 - (v2 & 0xFFFFFFF8) + 106) >> 3));
  return 0;
}
