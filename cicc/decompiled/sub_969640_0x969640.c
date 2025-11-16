// Function: sub_969640
// Address: 0x969640
//
bool __fastcall sub_969640(_QWORD *a1)
{
  _QWORD *v1; // rbx

  v1 = a1;
  if ( *a1 == sub_C33340() )
    v1 = (_QWORD *)a1[1];
  return (*((_BYTE *)v1 + 20) & 7) == 3;
}
