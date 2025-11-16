// Function: sub_867590
// Address: 0x867590
//
_QWORD *__fastcall sub_867590(_DWORD *a1)
{
  _QWORD *result; // rax

  if ( dword_4F04C44 != -1 )
  {
    result = qword_4F04C18;
    if ( qword_4F04C18 )
      goto LABEL_3;
    return (_QWORD *)sub_6851C0(0x780u, a1);
  }
  result = (_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64);
  if ( (*((_BYTE *)result + 6) & 2) == 0 )
    return result;
  result = qword_4F04C18;
  if ( !qword_4F04C18 )
    return (_QWORD *)sub_6851C0(0x780u, a1);
LABEL_3:
  if ( !*((_BYTE *)result + 42) && !result[2] )
  {
    result = (_QWORD *)result[1];
    *((_BYTE *)result + 60) = 1;
    result[4] = *(_QWORD *)a1;
  }
  return result;
}
