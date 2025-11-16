// Function: sub_1E641B0
// Address: 0x1e641b0
//
__int64 __fastcall sub_1E641B0(__int64 a1, _QWORD *a2)
{
  sub_1E64130(a2, *a2 & 0xFFFFFFFFFFFFFFF8LL);
  memset((void *)a1, 0, 0x80u);
  *(_DWORD *)(a1 + 24) = 8;
  *(_QWORD *)(a1 + 8) = a1 + 40;
  *(_QWORD *)(a1 + 16) = a1 + 40;
  return a1;
}
