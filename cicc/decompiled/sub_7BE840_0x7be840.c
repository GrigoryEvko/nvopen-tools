// Function: sub_7BE840
// Address: 0x7be840
//
__int64 __fastcall sub_7BE840(_DWORD *a1, __int64 *a2)
{
  unsigned int v2; // r12d
  __int64 v3; // rax
  __int64 v4; // rdx
  int v6; // r15d
  __int16 v7; // r14
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rax
  _BYTE v15[80]; // [rsp+10h] [rbp-50h] BYREF

  if ( dword_4D03D18 && (v2 = word_4F06418[0], word_4F06418[0] == 10) )
  {
    if ( a1 )
      *a1 = dword_4F06650[0];
    if ( a2 )
      *a2 = 0;
  }
  else
  {
    v3 = qword_4F08560;
    if ( !qword_4F08560 )
    {
      if ( !qword_4F08538 )
        goto LABEL_17;
      v3 = *(_QWORD *)(qword_4F08538 + 16);
      if ( !v3 )
        goto LABEL_17;
    }
    while ( *(_BYTE *)(v3 + 26) == 3 )
    {
      v3 = *(_QWORD *)v3;
      if ( !v3 )
        goto LABEL_17;
    }
    v2 = *(unsigned __int16 *)(v3 + 24);
    if ( (_WORD)v2 != 9 )
    {
      if ( a1 )
        *a1 = *(_DWORD *)(v3 + 28);
      if ( a2 )
      {
        v4 = 0;
        if ( (_WORD)v2 == 1 )
          v4 = *(_QWORD *)(v3 + 48);
        *a2 = v4;
      }
    }
    else
    {
LABEL_17:
      v6 = dword_4F07508[0];
      v7 = dword_4F07508[1];
      sub_7ADF70((__int64)v15, 0);
      sub_7AE360((__int64)v15);
      v2 = sub_7B8B50((unsigned __int64)v15, 0, v8, v9, v10, v11);
      if ( a1 )
        *a1 = dword_4F06650[0];
      if ( a2 )
      {
        v14 = 0;
        if ( (_WORD)v2 == 1 )
          v14 = qword_4D04A00;
        *a2 = v14;
      }
      sub_7BC000((unsigned __int64)v15, 0, (__int64)a1, (__int64)a2, v12, v13);
      dword_4F07508[0] = v6;
      LOWORD(dword_4F07508[1]) = v7;
    }
  }
  return v2;
}
