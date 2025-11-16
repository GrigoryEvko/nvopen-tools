// Function: sub_1776BC0
// Address: 0x1776bc0
//
__int64 __fastcall sub_1776BC0(__int64 a1)
{
  __int64 v1; // rax

  v1 = **(_QWORD **)(a1 - 24);
  if ( *(_BYTE *)(v1 + 8) == 16 )
    v1 = **(_QWORD **)(v1 + 16);
  return *(_DWORD *)(v1 + 8) >> 8;
}
