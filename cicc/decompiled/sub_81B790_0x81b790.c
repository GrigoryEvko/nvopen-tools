// Function: sub_81B790
// Address: 0x81b790
//
char *__fastcall sub_81B790(unsigned __int64 a1, __int64 a2)
{
  char *result; // rax
  __int64 v3; // r12
  __int64 *v4; // r13
  __int64 v5; // rsi

  result = (char *)qword_4F195B0 + qword_4F19598;
  if ( a1 >= (unsigned __int64)qword_4F195B0 + qword_4F19598 && qword_4F195A8 > a1 )
  {
    v3 = a2 - 1;
    v4 = sub_7AEFF0(a1);
    result = (char *)memchr((const void *)(a1 + 1), 10, a2 - 1);
    if ( result && (result = (char *)sub_7AF220((unsigned __int64)result), (_DWORD)result) )
    {
      v5 = qword_4F19590 - v4[5];
      v4[5] = v3;
    }
    else
    {
      v4[5] += v3;
      v5 = qword_4F19590;
    }
    qword_4F19590 = v5 + v3;
  }
  return result;
}
