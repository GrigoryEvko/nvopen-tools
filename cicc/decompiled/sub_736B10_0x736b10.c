// Function: sub_736B10
// Address: 0x736b10
//
_QWORD *__fastcall sub_736B10(char a1, __int64 a2)
{
  _QWORD *v2; // r12

  v2 = sub_7259C0(a1);
  if ( dword_4F077C4 == 2 )
    *((_BYTE *)v2 + 88) = v2[11] & 0x8F | 0x20;
  sub_87FA90(v2, a2);
  return v2;
}
