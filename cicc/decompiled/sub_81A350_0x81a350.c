// Function: sub_81A350
// Address: 0x81a350
//
_QWORD *__fastcall sub_81A350(_DWORD *a1, unsigned int *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r14
  unsigned __int64 v8; // rbx
  char v9; // di
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  _QWORD *v14; // r12
  __int64 v15; // r8
  __int64 v16; // r9

  v7 = 0;
  qword_4F06C40 = 0;
  if ( (unsigned __int16)sub_7B8B50((unsigned __int64)a1, a2, a3, a4, a5, a6) != 10 )
  {
    do
    {
      if ( word_4F06418[0] == 9 )
        break;
      if ( word_4F06418[0] == 28 )
      {
        if ( !v7 )
          break;
        --v7;
      }
      else
      {
        v7 += word_4F06418[0] == 27;
      }
      if ( unk_4F06400 )
      {
        v8 = 0;
        do
        {
          v9 = qword_4F06410[v8++];
          sub_729660(v9);
        }
        while ( unk_4F06400 > v8 );
      }
      sub_729660(32);
    }
    while ( (unsigned __int16)sub_7B8B50(0x20u, a2, v10, v11, v12, v13) != 10 );
  }
  sub_729660(0);
  v14 = qword_4F06C50;
  if ( !(unsigned int)sub_7BE280(0x1Cu, 18, 0, 0, v15, v16) )
    *a1 = 1;
  return v14;
}
