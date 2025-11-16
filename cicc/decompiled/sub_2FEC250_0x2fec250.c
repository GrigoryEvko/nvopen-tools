// Function: sub_2FEC250
// Address: 0x2fec250
//
__int64 __fastcall sub_2FEC250(__int64 a1, char a2)
{
  __int64 (*v2)(); // rdx
  __int64 result; // rax
  unsigned int *v4; // rax

  if ( BYTE4(qword_4F86250[2]) )
  {
    v4 = (unsigned int *)sub_C94E20((__int64)qword_4F86250);
    if ( !v4 )
    {
      result = LODWORD(qword_4F86250[2]);
LABEL_3:
      if ( a2 )
        return (unsigned int)qword_5026BA8;
      return result;
    }
    result = *v4;
    if ( a2 )
      return (unsigned int)qword_5026BA8;
  }
  else
  {
    v2 = *(__int64 (**)())(*(_QWORD *)a1 + 1840LL);
    result = 10;
    if ( v2 == sub_2FE36A0 )
      goto LABEL_3;
    result = v2();
    if ( a2 )
      return (unsigned int)qword_5026BA8;
  }
  return result;
}
