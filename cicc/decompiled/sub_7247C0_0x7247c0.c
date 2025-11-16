// Function: sub_7247C0
// Address: 0x7247c0
//
_QWORD *__fastcall sub_7247C0(__int64 a1)
{
  _QWORD *v1; // rax
  _QWORD *result; // rax

  v1 = (_QWORD *)(dword_4F07988 + sub_822F50(unk_4F073B8, a1 + dword_4F0798C));
  if ( !unk_4D03FE8 )
    *v1++ = 0;
  *v1 = 0;
  result = v1 + 2;
  *((_BYTE *)result - 8) = (2 * (unk_4D03FE8 == 0)) | (8 * (unk_4F06CFC & 1)) | 1;
  return result;
}
