// Function: sub_1704F50
// Address: 0x1704f50
//
__int64 __fastcall sub_1704F50(__int64 a1)
{
  __int64 v1; // rax

  v1 = **(_QWORD **)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
  if ( *(_BYTE *)(v1 + 8) == 16 )
    v1 = **(_QWORD **)(v1 + 16);
  return *(_DWORD *)(v1 + 8) >> 8;
}
