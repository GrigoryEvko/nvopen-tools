// Function: sub_2553790
// Address: 0x2553790
//
__int64 __fastcall sub_2553790(const char **a1, const char *a2)
{
  size_t v2; // rbx
  int v3; // eax
  unsigned int v4; // r14d
  __int64 result; // rax
  __int64 v6; // rax
  _QWORD *v7; // rcx
  _QWORD *v8; // r15
  __int64 v9; // [rsp+8h] [rbp-38h]

  *a1 = a2;
  if ( a2 )
  {
    a1[1] = (const char *)strlen(a2);
    v2 = strlen(a2);
  }
  else
  {
    a1[1] = 0;
    v2 = 0;
  }
  v3 = sub_C92610();
  v4 = sub_C92740((__int64)qword_5032AD0, a2, v2, v3);
  result = *(_QWORD *)(qword_5032AD0[0] + 8LL * v4);
  if ( result )
  {
    if ( result != -8 )
      return result;
    --LODWORD(qword_5032AD0[2]);
  }
  v9 = qword_5032AD0[0] + 8LL * v4;
  v6 = sub_C7D670(v2 + 9, 8);
  v7 = (_QWORD *)v9;
  v8 = (_QWORD *)v6;
  if ( v2 )
  {
    memcpy((void *)(v6 + 8), a2, v2);
    v7 = (_QWORD *)v9;
  }
  *((_BYTE *)v8 + v2 + 8) = 0;
  *v8 = v2;
  *v7 = v8;
  ++HIDWORD(qword_5032AD0[1]);
  return sub_C929D0(qword_5032AD0, v4);
}
