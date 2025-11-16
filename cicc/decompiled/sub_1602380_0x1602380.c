// Function: sub_1602380
// Address: 0x1602380
//
char __fastcall sub_1602380(__int64 a1)
{
  char result; // al
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v4; // rbx
  _QWORD *v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rcx
  _QWORD *v8; // rcx
  __int64 v9; // rsi
  __int64 v10; // rsi
  _QWORD *v11; // rcx
  __int64 v12; // rsi
  _QWORD *v13; // rdx
  _QWORD *v14; // rdx
  _QWORD *v15; // rdx

  result = 1;
  if ( *(char *)(a1 + 23) < 0 )
  {
    v2 = sub_1648A40(a1);
    v4 = v2 + v3;
    if ( *(char *)(a1 + 23) >= 0 )
    {
      v5 = 0;
      v7 = v4 >> 4;
      v6 = v4 >> 6;
    }
    else
    {
      v5 = (_QWORD *)sub_1648A40(a1);
      v6 = (v4 - (__int64)v5) >> 6;
      v7 = (v4 - (__int64)v5) >> 4;
    }
    if ( v6 > 0 )
    {
      while ( 1 )
      {
        v8 = (_QWORD *)*v5;
        if ( *(_QWORD *)*v5 != 6 || *((_DWORD *)v8 + 4) != 1869506409 || *((_WORD *)v8 + 10) != 25970 )
          return v4 == (_QWORD)v5;
        v9 = v5[2];
        if ( *(_QWORD *)v9 != 6 || *(_DWORD *)(v9 + 16) != 1869506409 || *(_WORD *)(v9 + 20) != 25970 )
          return v5 + 2 == (_QWORD *)v4;
        v10 = v5[4];
        v11 = v5 + 4;
        if ( *(_QWORD *)v10 != 6 )
          return v4 == (_QWORD)v11;
        if ( *(_DWORD *)(v10 + 16) != 1869506409 )
          return v4 == (_QWORD)v11;
        if ( *(_WORD *)(v10 + 20) != 25970 )
          return v4 == (_QWORD)v11;
        v12 = v5[6];
        v11 = v5 + 6;
        if ( *(_QWORD *)v12 != 6 || *(_DWORD *)(v12 + 16) != 1869506409 || *(_WORD *)(v12 + 20) != 25970 )
          return v4 == (_QWORD)v11;
        v5 += 8;
        if ( !--v6 )
        {
          v7 = (v4 - (__int64)v5) >> 4;
          break;
        }
      }
    }
    if ( v7 != 2 )
    {
      if ( v7 != 3 )
      {
        if ( v7 != 1 )
          return 1;
        goto LABEL_35;
      }
      v14 = (_QWORD *)*v5;
      if ( *(_QWORD *)*v5 != 6 || *((_DWORD *)v14 + 4) != 1869506409 || *((_WORD *)v14 + 10) != 25970 )
        return v4 == (_QWORD)v5;
      v5 += 2;
    }
    v13 = (_QWORD *)*v5;
    if ( *(_QWORD *)*v5 != 6 || *((_DWORD *)v13 + 4) != 1869506409 || *((_WORD *)v13 + 10) != 25970 )
      return v4 == (_QWORD)v5;
    v5 += 2;
LABEL_35:
    v15 = (_QWORD *)*v5;
    if ( *(_QWORD *)*v5 == 6 && *((_DWORD *)v15 + 4) == 1869506409 && *((_WORD *)v15 + 10) == 25970 )
      return 1;
    return v4 == (_QWORD)v5;
  }
  return result;
}
