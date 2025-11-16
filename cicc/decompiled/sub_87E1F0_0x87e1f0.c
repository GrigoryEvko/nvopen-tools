// Function: sub_87E1F0
// Address: 0x87e1f0
//
void __fastcall sub_87E1F0(__int64 a1)
{
  _QWORD *v1; // rax
  _QWORD *v2; // rdx

  if ( qword_4F5FFE0 )
  {
    if ( !a1 )
      return;
    v1 = (_QWORD *)a1;
    do
    {
      v2 = v1;
      v1 = (_QWORD *)*v1;
    }
    while ( v1 );
    *v2 = qword_4F5FFE0;
  }
  qword_4F5FFE0 = a1;
}
