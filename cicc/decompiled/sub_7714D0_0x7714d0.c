// Function: sub_7714D0
// Address: 0x7714d0
//
_QWORD *__fastcall sub_7714D0(__int64 a1)
{
  _QWORD *v1; // r12
  _QWORD *v2; // rbx
  __int64 v3; // rdx
  _QWORD *result; // rax
  __int64 v5; // rdx

  v1 = (_QWORD *)(a1 + 16);
  v2 = *(_QWORD **)(a1 + 16);
  do
  {
    result = (_QWORD *)qword_4F08088;
    if ( qword_4F08088 )
    {
      v3 = *(_QWORD *)qword_4F08088;
      --qword_4F08080;
      qword_4F08088 = v3;
    }
    else
    {
      result = (_QWORD *)sub_823970(32);
      v5 = qword_4F08098;
      ++qword_4F08090;
      qword_4F08098 = (__int64)result;
      result[1] = v5;
    }
    *v1 = result;
    v1 = result;
    result[2] = v2[2];
    result[3] = v2[3];
    v2 = (_QWORD *)*v2;
  }
  while ( v2 );
  *result = 0;
  return result;
}
