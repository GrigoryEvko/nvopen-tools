// Function: sub_157F280
// Address: 0x157f280
//
__int64 __fastcall sub_157F280(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // r8

  if ( a1 + 40 == (*(_QWORD *)(a1 + 40) & 0xFFFFFFFFFFFFFFF8LL) )
    return 0;
  v1 = *(_QWORD *)(a1 + 48);
  if ( !v1 )
    BUG();
  v2 = 0;
  if ( *(_BYTE *)(v1 - 8) == 77 )
    return v1 - 24;
  return v2;
}
