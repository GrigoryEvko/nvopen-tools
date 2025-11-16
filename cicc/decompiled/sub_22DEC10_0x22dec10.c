// Function: sub_22DEC10
// Address: 0x22dec10
//
__int64 __fastcall sub_22DEC10(__int64 a1, _QWORD *a2)
{
  sub_22DE030(a2, *a2 & 0xFFFFFFFFFFFFFFF8LL);
  memset((void *)a1, 0, 0x78u);
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_DWORD *)(a1 + 16) = 8;
  return a1;
}
