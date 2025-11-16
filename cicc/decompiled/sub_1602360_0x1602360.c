// Function: sub_1602360
// Address: 0x1602360
//
bool __fastcall sub_1602360(__int64 a1)
{
  __int64 v1; // rax

  v1 = *(_QWORD *)(a1 - 24);
  if ( *(_BYTE *)(v1 + 16) )
    BUG();
  return *(_DWORD *)(v1 + 36) == 62;
}
