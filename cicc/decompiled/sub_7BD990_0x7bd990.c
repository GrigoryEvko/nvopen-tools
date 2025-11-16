// Function: sub_7BD990
// Address: 0x7bd990
//
_BOOL8 sub_7BD990()
{
  unsigned __int8 v0; // bl
  int v1; // r13d
  int v2; // eax
  _BOOL8 result; // rax
  unsigned __int8 *v4; // rax

  v0 = *qword_4F06460;
  if ( *qword_4F06460 == 9 || v0 == 32 )
  {
    v4 = qword_4F06460 + 1;
    do
    {
      do
      {
        qword_4F06460 = v4;
        v0 = *v4++;
      }
      while ( v0 == 9 );
    }
    while ( v0 == 32 );
  }
  v1 = v0;
  v2 = iscntrl(v0);
  if ( v0 == 47 || v2 )
  {
    sub_7BC390();
    v1 = (unsigned __int8)*qword_4F06460;
    v0 = *qword_4F06460;
  }
  result = 1;
  if ( (unsigned int)(v1 - 48) > 9 )
  {
    if ( !dword_4F055C0[(char)*qword_4F06460 + 128] && (unsigned int)sub_7B3CF0(qword_4F06460, 0, 1) )
      return 0;
    if ( v0 == 58 )
    {
      if ( qword_4F06460[1] == 58 )
        return 0;
    }
    else if ( v0 == 60 || v0 == 46 && HIDWORD(qword_4D0495C) )
    {
      return 0;
    }
    return v0 != 35;
  }
  return result;
}
