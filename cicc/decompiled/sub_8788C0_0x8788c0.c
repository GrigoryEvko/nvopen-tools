// Function: sub_8788C0
// Address: 0x8788c0
//
void __fastcall sub_8788C0(__int64 a1)
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
    *v2 = qword_4F5FFF0;
    qword_4F5FFF0 = a1;
  }
}
