// Function: sub_67B070
// Address: 0x67b070
//
__int64 __fastcall sub_67B070(_DWORD *a1, unsigned __int16 a2, __int64 a3, __int64 a4)
{
  unsigned int v4; // r13d
  int v6; // esi
  __int64 v7; // rcx
  __int64 v8; // rdx
  unsigned __int16 v9; // ax
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 result; // rax
  __int64 v14; // r15
  __int64 v15; // rsi
  __int64 v16; // rdi
  __int64 v17; // rdx
  __int64 v18; // rcx
  unsigned __int16 v19; // r14
  __int64 v20; // rcx
  _QWORD v21[7]; // [rsp+8h] [rbp-38h] BYREF

  v4 = a2;
  if ( word_4F06418[0] != 28 )
  {
    while ( 1 )
    {
      v6 = a1[11];
      v21[0] = 0;
      if ( v6 )
        break;
      if ( (unsigned int)sub_869470(v21) )
        break;
LABEL_17:
      if ( word_4F06418[0] == 28 )
        goto LABEL_18;
    }
    while ( 1 )
    {
      sub_6799B0(v4);
      if ( word_4F06418[0] == 76 )
      {
        sub_679930(v4, 0, v10, v11);
        v9 = word_4F06418[0];
        if ( word_4F06418[0] == 67 )
          goto LABEL_12;
      }
      else
      {
        sub_67A900(a1, unk_4D04874 == 0 ? 7 : 2055, 0, v11);
        v7 = (unsigned int)a1[3];
        if ( !(_DWORD)v7 )
          return sub_867030(v21[0]);
        v8 = (unsigned int)a1[4];
        if ( (_DWORD)v8 )
          return sub_867030(v21[0]);
        v9 = word_4F06418[0];
        if ( word_4F06418[0] == 67 )
        {
LABEL_12:
          sub_679930(v4, 0, v8, v7);
          goto LABEL_9;
        }
      }
      if ( v9 != 76 && v9 != 28 )
      {
        a1[3] = 0;
        return sub_867030(v21[0]);
      }
LABEL_9:
      sub_867630(v21[0], 1);
      if ( !(unsigned int)sub_866C00(v21[0]) )
        goto LABEL_17;
    }
  }
LABEL_18:
  v14 = 0x6004000001LL;
  v15 = 0;
  v16 = v4;
  sub_679930(v4, 0, a3, a4);
  while ( 1 )
  {
    v19 = word_4F06418[0];
    if ( (unsigned __int16)(word_4F06418[0] - 81) > 0x26u )
      break;
    if ( !_bittest64(&v14, (unsigned int)word_4F06418[0] - 81) )
      goto LABEL_26;
LABEL_20:
    v15 = 0;
    v16 = v4;
    sub_679930(v4, 0, v17, v18);
  }
  if ( (unsigned __int16)(word_4F06418[0] - 263) <= 3u )
    goto LABEL_20;
  if ( word_4F06418[0] == 33 || word_4F06418[0] == 52 )
  {
    v15 = 0;
    v16 = v4;
    sub_679930(v4, 0, v17, v18);
    v19 = word_4F06418[0];
  }
LABEL_26:
  LOBYTE(v17) = v19 == 281 || v19 == 162;
  if ( (_BYTE)v17 || v19 == 243 )
  {
    v15 = 0;
    v16 = v4;
    result = (__int64)sub_679930(v4, 0, v17, v18);
    if ( word_4F06418[0] == 27 )
    {
      sub_679930(v4, 0, v17, v18);
      v15 = 1;
      v16 = 28;
      sub_6797D0(0x1Cu, 1);
      if ( word_4F06418[0] == 28 )
      {
        v15 = 0;
        v16 = v4;
        sub_679930(v4, 0, v17, v18);
      }
      goto LABEL_28;
    }
    if ( v19 == 243 )
      goto LABEL_28;
    a1[3] = 0;
  }
  else
  {
LABEL_28:
    result = word_4D04430;
    if ( word_4D04430 )
    {
      if ( word_4F06418[0] == 30 )
      {
        sub_7B8B50(v16, v15, v17, v18);
        return sub_67A900(a1, 1, 0, v20);
      }
    }
  }
  return result;
}
