// Function: sub_1D917F0
// Address: 0x1d917f0
//
_QWORD **__fastcall sub_1D917F0(_QWORD *a1)
{
  _QWORD **result; // rax

  result = &qword_4FC3618;
  if ( qword_4FC3618 )
    *qword_4FC3618 = a1;
  else
    unk_4FC3620 = a1;
  qword_4FC3618 = a1;
  return result;
}
