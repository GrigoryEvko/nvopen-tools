// Function: sub_1095380
// Address: 0x1095380
//
__int64 __fastcall sub_1095380(_QWORD *a1)
{
  _BYTE *v1; // rdx
  __int64 result; // rax
  int v3; // eax

  v1 = (_BYTE *)*a1;
  result = *(_BYTE *)*a1 & 0xDF;
  if ( (*(_BYTE *)*a1 & 0xDF) == 0x55 )
  {
    *a1 = v1 + 1;
    v3 = (unsigned __int8)*++v1;
    result = v3 & 0xFFFFFFDF;
  }
  if ( (_BYTE)result == 76 )
  {
    *a1 = v1 + 1;
    result = v1[1] & 0xDF;
    if ( (v1[1] & 0xDF) == 0x4C )
      *a1 = v1 + 2;
  }
  return result;
}
