// Function: sub_8CCD60
// Address: 0x8ccd60
//
void sub_8CCD60()
{
  __int64 i; // r12
  _QWORD *v1; // rbx
  __int64 v2; // rdi
  char v3; // al

  for ( i = qword_4F60240; qword_4F60240; i = qword_4F60240 )
  {
    qword_4F60240 = 0;
    v1 = (_QWORD *)i;
    do
    {
      while ( 1 )
      {
        v2 = v1[1];
        if ( v2 )
          break;
LABEL_8:
        v1 = (_QWORD *)*v1;
        if ( !v1 )
          goto LABEL_12;
      }
      v3 = *(_BYTE *)(v2 + 80);
      if ( (unsigned __int8)(v3 - 4) > 1u )
      {
        if ( (unsigned __int8)(v3 - 10) <= 1u || v3 == 17 )
        {
          sub_8CC1D0(v2);
        }
        else if ( v3 == 3 )
        {
          sub_8C9210(v2);
        }
        else if ( v3 == 7 )
        {
          sub_8CC330(v2);
        }
        goto LABEL_8;
      }
      sub_8CCC20(v2);
      v1 = (_QWORD *)*v1;
    }
    while ( v1 );
LABEL_12:
    sub_878490(i);
  }
}
