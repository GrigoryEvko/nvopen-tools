// Function: sub_27574D0
// Address: 0x27574d0
//
_QWORD *__fastcall sub_27574D0(_QWORD *a1, __int64 *a2, __int64 a3)
{
  _QWORD *v4; // rax
  __int64 v6; // r9
  _QWORD *v7; // rdx
  __int64 v8; // rcx

  v4 = (_QWORD *)a2[1];
  v6 = *a2;
  if ( !*((_BYTE *)a2 + 28) )
  {
    v7 = &v4[*((unsigned int *)a2 + 4)];
    goto LABEL_5;
  }
  v7 = &v4[*((unsigned int *)a2 + 5)];
  while ( v7 != v4 )
  {
    if ( *v4 < 0xFFFFFFFFFFFFFFFELL )
      break;
    ++v4;
LABEL_5:
    ;
  }
  v8 = a3 - 1;
  if ( a3 )
  {
    do
    {
      for ( ++v4; v7 != v4; ++v4 )
      {
        if ( *v4 < 0xFFFFFFFFFFFFFFFELL )
          break;
      }
    }
    while ( v8-- != 0 );
  }
  *a1 = v4;
  a1[1] = v7;
  a1[2] = a2;
  a1[3] = v6;
  a1[4] = v7;
  a1[5] = v7;
  a1[6] = a2;
  a1[7] = v6;
  return a1;
}
