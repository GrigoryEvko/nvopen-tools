// Function: sub_947280
// Address: 0x947280
//
_QWORD *__fastcall sub_947280(__int64 a1)
{
  _QWORD *result; // rax

  result = *(_QWORD **)(a1 + 8);
  if ( !result )
  {
    result = (_QWORD *)sub_22077B0(8);
    if ( result )
      *result = &unk_49D94A8;
    *(_QWORD *)(a1 + 8) = result;
  }
  return result;
}
