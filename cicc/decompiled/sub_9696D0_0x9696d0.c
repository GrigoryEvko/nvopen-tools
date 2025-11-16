// Function: sub_9696D0
// Address: 0x9696d0
//
_BOOL8 __fastcall sub_9696D0(_QWORD *a1)
{
  _QWORD *v1; // rbx

  v1 = a1;
  if ( *a1 == sub_C33340() )
    v1 = (_QWORD *)a1[1];
  return (*((_BYTE *)v1 + 20) & 8) != 0;
}
