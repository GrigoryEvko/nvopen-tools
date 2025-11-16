// Function: sub_878490
// Address: 0x878490
//
void __fastcall sub_878490(__int64 a1)
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
    *v2 = qword_4F60000;
    qword_4F60000 = a1;
  }
}
