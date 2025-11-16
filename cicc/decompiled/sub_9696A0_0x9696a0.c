// Function: sub_9696A0
// Address: 0x9696a0
//
bool __fastcall sub_9696A0(_QWORD *a1)
{
  _QWORD *v1; // rbx

  v1 = a1;
  if ( *a1 == sub_C33340() )
    v1 = (_QWORD *)a1[1];
  return (*((_BYTE *)v1 + 20) & 7) == 1;
}
