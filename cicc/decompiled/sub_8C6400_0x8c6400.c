// Function: sub_8C6400
// Address: 0x8c6400
//
_QWORD *__fastcall sub_8C6400(char a1, __int64 a2)
{
  _QWORD *v2; // rax
  _QWORD *v3; // r12
  __int64 v5; // rdx

  if ( a1 == 37 )
  {
    v2 = *(_QWORD **)(a2 + 64);
    v3 = (_QWORD *)(a2 + 64);
  }
  else
  {
    v2 = *(_QWORD **)(a2 + 32);
    v3 = (_QWORD *)(a2 + 32);
  }
  if ( v2 )
  {
    if ( *v2 != a2 )
    {
LABEL_5:
      sub_8D0810(*v3);
      *v3 = 0;
      return v3;
    }
    v5 = v2[1];
    if ( v5 && v5 != a2 )
    {
      *v2 = v5;
      goto LABEL_5;
    }
  }
  return v3;
}
