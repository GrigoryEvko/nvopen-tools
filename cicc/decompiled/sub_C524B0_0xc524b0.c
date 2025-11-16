// Function: sub_C524B0
// Address: 0xc524b0
//
_QWORD *__fastcall sub_C524B0(__int64 a1)
{
  __int64 v2; // rdi
  _QWORD *result; // rax
  __int64 v4; // rcx
  _QWORD *v5; // rdx

  if ( !qword_4F83CE0 )
    sub_C7D570(&qword_4F83CE0, sub_C53DA0, sub_C50EC0);
  v2 = qword_4F83CE0;
  if ( !*(_BYTE *)(qword_4F83CE0 + 148) )
    return (_QWORD *)sub_C8CC70(qword_4F83CE0 + 120, a1);
  result = *(_QWORD **)(qword_4F83CE0 + 128);
  v4 = *(unsigned int *)(qword_4F83CE0 + 140);
  v5 = &result[v4];
  if ( result == v5 )
  {
LABEL_9:
    if ( (unsigned int)v4 >= *(_DWORD *)(qword_4F83CE0 + 136) )
      return (_QWORD *)sub_C8CC70(qword_4F83CE0 + 120, a1);
    *(_DWORD *)(qword_4F83CE0 + 140) = v4 + 1;
    *v5 = a1;
    ++*(_QWORD *)(v2 + 120);
  }
  else
  {
    while ( a1 != *result )
    {
      if ( v5 == ++result )
        goto LABEL_9;
    }
  }
  return result;
}
