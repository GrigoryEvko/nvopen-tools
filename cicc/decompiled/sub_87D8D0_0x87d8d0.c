// Function: sub_87D8D0
// Address: 0x87d8d0
//
__int64 __fastcall sub_87D8D0(_QWORD *a1)
{
  _QWORD *v1; // r13
  _QWORD *v2; // r12
  _QWORD *i; // rbx
  __int64 v4; // rax

  if ( !a1 )
    return 0;
  v1 = a1;
  do
  {
    v2 = (_QWORD *)v1[2];
    for ( i = (_QWORD *)v1[1]; (_QWORD *)*v2 != i; i = (_QWORD *)*i )
    {
      v4 = i[2];
      if ( i == v2 || (*(_BYTE *)(v4 + 96) & 2) == 0 )
      {
        if ( (unsigned int)sub_87D890(*(_QWORD *)(v4 + 40)) )
          return 1;
      }
      else if ( (unsigned int)sub_87D8D0(*(_QWORD *)(v4 + 112)) )
      {
        return 1;
      }
    }
    v1 = (_QWORD *)*v1;
  }
  while ( v1 );
  return 0;
}
