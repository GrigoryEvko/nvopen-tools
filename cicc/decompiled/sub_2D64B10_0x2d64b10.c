// Function: sub_2D64B10
// Address: 0x2d64b10
//
_QWORD *__fastcall sub_2D64B10(_QWORD *a1, __int64 *a2, _QWORD *a3)
{
  __int64 v4; // rax
  _QWORD *v6; // rcx
  char v7; // si
  __int64 v8; // rax
  __int64 v10; // rax

  v4 = a2[1];
  if ( *((_BYTE *)a2 + 28) )
    v6 = (_QWORD *)(v4 + 8LL * *((unsigned int *)a2 + 5));
  else
    v6 = (_QWORD *)(v4 + 8LL * *((unsigned int *)a2 + 4));
  *a1 = a3;
  a1[1] = v6;
  if ( a3 == v6 )
  {
LABEL_8:
    v8 = *a2;
    a1[2] = a2;
    a1[3] = v8;
    return a1;
  }
  else
  {
    v7 = 0;
    while ( *a3 == -2 || *a3 == -1 )
    {
      ++a3;
      v7 = 1;
      if ( a3 == v6 )
        goto LABEL_7;
    }
    if ( v7 )
    {
LABEL_7:
      *a1 = a3;
      goto LABEL_8;
    }
    v10 = *a2;
    a1[2] = a2;
    a1[3] = v10;
    return a1;
  }
}
