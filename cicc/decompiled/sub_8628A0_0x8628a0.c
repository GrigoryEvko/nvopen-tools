// Function: sub_8628A0
// Address: 0x8628a0
//
__int64 sub_8628A0()
{
  __int64 result; // rax

  result = qword_4F5FCE8;
  if ( qword_4F5FCE8 )
  {
    do
    {
      sub_862730(*(_DWORD *)(*(_QWORD *)(result + 8) + 160LL), 0);
      result = *(_QWORD *)qword_4F5FCE8;
      qword_4F5FCE8 = result;
    }
    while ( result );
    qword_4F5FCE0 = 0;
  }
  else
  {
    qword_4F5FCE0 = 0;
  }
  return result;
}
