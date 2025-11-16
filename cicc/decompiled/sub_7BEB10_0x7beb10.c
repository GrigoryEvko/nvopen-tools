// Function: sub_7BEB10
// Address: 0x7beb10
//
__int64 __fastcall sub_7BEB10(unsigned int a1, _WORD *a2)
{
  unsigned int v2; // r12d
  __int64 v3; // rax
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  unsigned int v13; // r12d
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  __int16 v18; // ax
  _BYTE v19[64]; // [rsp+0h] [rbp-40h] BYREF

  if ( dword_4D03D18 )
  {
    v2 = word_4F06418[0];
    if ( word_4F06418[0] == 10 )
      goto LABEL_9;
  }
  v3 = qword_4F08560;
  if ( qword_4F08560 || qword_4F08538 && (v3 = *(_QWORD *)(qword_4F08538 + 16)) != 0 )
  {
    while ( *(_BYTE *)(v3 + 26) == 3 )
    {
      v3 = *(_QWORD *)v3;
      if ( !v3 )
        goto LABEL_12;
    }
    v2 = *(unsigned __int16 *)(v3 + 24);
    if ( (_WORD)v2 != 9 )
    {
      if ( (_WORD)v2 != (_WORD)a1 )
      {
LABEL_9:
        *a2 = 0;
        return v2;
      }
      while ( 1 )
      {
        v3 = *(_QWORD *)v3;
        if ( !v3 )
          break;
        if ( *(_BYTE *)(v3 + 26) != 3 )
        {
          v18 = *(_WORD *)(v3 + 24);
          if ( v18 == 9 )
            break;
          *a2 = v18;
          return a1;
        }
      }
    }
  }
LABEL_12:
  sub_7ADF70((__int64)v19, 0);
  sub_7AE360((__int64)v19);
  v13 = sub_7B8B50((unsigned __int64)v19, 0, v5, v6, v7, v8);
  if ( (_WORD)a1 == (_WORD)v13 )
  {
    sub_7AE360((__int64)v19);
    *a2 = sub_7B8B50((unsigned __int64)v19, 0, v14, v15, v16, v17);
  }
  else
  {
    *a2 = 0;
  }
  sub_7BC000((unsigned __int64)v19, 0, v9, v10, v11, v12);
  return v13;
}
