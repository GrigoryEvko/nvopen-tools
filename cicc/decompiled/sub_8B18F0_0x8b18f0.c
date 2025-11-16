// Function: sub_8B18F0
// Address: 0x8b18f0
//
void sub_8B18F0()
{
  _QWORD *v0; // r14
  _QWORD *v1; // rbx
  __int64 v2; // r15
  _QWORD *v3; // rbx
  __int64 v4; // rdi

  v0 = qword_4D03FD0;
  if ( qword_4D03FD0 )
  {
    do
    {
      sub_8D0A80(v0);
      v1 = (_QWORD *)qword_4F60208;
      if ( qword_4F60208 )
      {
        do
        {
          while ( 1 )
          {
            if ( dword_4F077C4 == 2 )
            {
              v2 = v1[1];
              if ( (unsigned int)sub_8D23B0(v2) )
              {
                if ( (unsigned int)sub_8D3A70(v2) )
                  break;
              }
            }
            v1 = (_QWORD *)*v1;
            if ( !v1 )
              goto LABEL_9;
          }
          sub_8AD220(v2, 0);
          v1 = (_QWORD *)*v1;
        }
        while ( v1 );
      }
LABEL_9:
      dword_4F601E0 = 1;
      sub_8B17A0();
      sub_8D0B10();
      v0 = (_QWORD *)*v0;
    }
    while ( v0 );
    do
    {
      dword_4F601C8 = 0;
      v3 = qword_4D03FD0;
      if ( !qword_4D03FD0 )
        break;
      do
      {
        if ( *((_BYTE *)v3 + 385) )
        {
          *((_BYTE *)v3 + 385) = 0;
          sub_8D0A80(v3);
          sub_8B17A0();
          v4 = qword_4F07288;
          nullsub_9();
          sub_5EB2E0(v4);
          sub_8D0B10();
        }
        v3 = (_QWORD *)*v3;
      }
      while ( v3 );
    }
    while ( dword_4F601C8 );
  }
  else
  {
    dword_4F601C8 = 0;
  }
}
