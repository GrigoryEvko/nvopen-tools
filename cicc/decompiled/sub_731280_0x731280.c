// Function: sub_731280
// Address: 0x731280
//
_BYTE *__fastcall sub_731280(__int64 a1)
{
  _BYTE *result; // rax
  __int64 v2; // rdx

  if ( (*(_BYTE *)(a1 + 207) & 0x30) == 0x10 )
    sub_8B1A30(a1, dword_4F07508);
  result = sub_726700(20);
  v2 = *(_QWORD *)(a1 + 152);
  result[25] |= 1u;
  *(_QWORD *)result = v2;
  *((_QWORD *)result + 7) = a1;
  return result;
}
