// Function: sub_22109F0
// Address: 0x22109f0
//
void __fastcall sub_22109F0(_QWORD *a1)
{
  _QWORD *v1; // rbx
  _QWORD *v2; // rax

  if ( a1 )
  {
    v1 = a1;
    do
    {
      v2 = v1;
      v1 = (_QWORD *)*v1;
      ((void (__fastcall *)(_QWORD *))v2[1])(v2);
    }
    while ( v1 );
  }
}
