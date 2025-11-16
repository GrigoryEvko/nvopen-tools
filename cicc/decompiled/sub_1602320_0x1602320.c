// Function: sub_1602320
// Address: 0x1602320
//
bool __fastcall sub_1602320(__int64 a1)
{
  __int64 v1; // rax
  unsigned int v2; // ecx

  v1 = *(_QWORD *)(a1 - 24);
  if ( *(_BYTE *)(v1 + 16) )
    BUG();
  v2 = *(_DWORD *)(v1 + 36) - 57;
  return v2 <= 0x11 && ((1LL << v2) & 0x39E07) != 0;
}
