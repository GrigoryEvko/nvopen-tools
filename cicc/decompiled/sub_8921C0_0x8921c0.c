// Function: sub_8921C0
// Address: 0x8921c0
//
void __fastcall sub_8921C0(__int64 a1)
{
  _QWORD *v1; // rax
  _QWORD *v2; // rdx

  if ( a1 )
  {
    v1 = (_QWORD *)a1;
    do
    {
      v2 = v1;
      v1 = (_QWORD *)*v1;
    }
    while ( v1 );
    *v2 = qword_4F601A0;
    qword_4F601A0 = a1;
  }
}
