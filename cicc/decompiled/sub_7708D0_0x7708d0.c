// Function: sub_7708D0
// Address: 0x7708d0
//
__int64 sub_7708D0()
{
  __int64 result; // rax
  __int64 v1; // rdx
  __int64 v2; // rdx

  result = qword_4F08088;
  if ( qword_4F08088 )
  {
    v1 = *(_QWORD *)qword_4F08088;
    --qword_4F08080;
    qword_4F08088 = v1;
  }
  else
  {
    result = sub_823970(32);
    v2 = qword_4F08098;
    ++qword_4F08090;
    qword_4F08098 = result;
    *(_QWORD *)(result + 8) = v2;
  }
  return result;
}
