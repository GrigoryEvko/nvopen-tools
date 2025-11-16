// Function: sub_6E1940
// Address: 0x6e1940
//
__int64 __fastcall sub_6E1940(_QWORD *a1)
{
  _QWORD *v1; // rbx
  _QWORD *v2; // r12
  __int64 result; // rax

  if ( a1 )
  {
    v1 = a1;
    do
    {
      v2 = v1;
      v1 = (_QWORD *)*v1;
      nullsub_5();
      result = qword_4D03A88;
      *v2 = qword_4D03A88;
      qword_4D03A88 = (__int64)v2;
    }
    while ( v1 );
  }
  return result;
}
