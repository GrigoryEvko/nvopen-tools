// Function: sub_193E280
// Address: 0x193e280
//
__int64 __fastcall sub_193E280(__int64 a1)
{
  __int64 v1; // rax
  unsigned __int64 v2; // rax
  __int64 result; // rax

  if ( !sub_13FCB50(a1) )
    return 0;
  v1 = sub_13F9E70(a1);
  v2 = sub_157EBA0(v1);
  if ( *(_BYTE *)(v2 + 16) != 26 )
    BUG();
  result = *(_QWORD *)(v2 - 72);
  if ( *(_BYTE *)(result + 16) != 75 )
    return 0;
  return result;
}
