// Function: sub_16B5AB0
// Address: 0x16b5ab0
//
_QWORD *__fastcall sub_16B5AB0(_QWORD *a1, __int64 *a2, _QWORD *a3)
{
  __int64 v4; // rax
  _QWORD *v6; // rcx
  char v7; // si
  __int64 v8; // rax
  __int64 v10; // rax

  v4 = a2[2];
  if ( v4 == a2[1] )
    v6 = (_QWORD *)(v4 + 8LL * *((unsigned int *)a2 + 7));
  else
    v6 = (_QWORD *)(v4 + 8LL * *((unsigned int *)a2 + 6));
  *a1 = a3;
  a1[1] = v6;
  if ( a3 == v6 )
    goto LABEL_8;
  v7 = 0;
  while ( *a3 == -2 || *a3 == -1 )
  {
    ++a3;
    v7 = 1;
    if ( v6 == a3 )
    {
      *a1 = v6;
      goto LABEL_8;
    }
  }
  if ( !v7 )
  {
LABEL_8:
    v8 = *a2;
    a1[2] = a2;
    a1[3] = v8;
    return a1;
  }
  else
  {
    v10 = *a2;
    *a1 = a3;
    a1[2] = a2;
    a1[3] = v10;
    return a1;
  }
}
