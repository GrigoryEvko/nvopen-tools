// Function: sub_25AFE60
// Address: 0x25afe60
//
__int64 __fastcall sub_25AFE60(_QWORD *a1, __int64 *a2)
{
  _QWORD *v2; // rax
  _QWORD *v5; // r10
  unsigned __int64 v6; // rsi
  _QWORD *v7; // rdi
  __int64 v8; // rcx
  __int64 v9; // rdx
  _QWORD *v10; // rax
  unsigned __int64 v11; // rdx
  _QWORD *v12; // rcx
  unsigned int v13; // edi
  unsigned int v14; // eax

  v2 = (_QWORD *)a1[14];
  v5 = a1 + 13;
  if ( v2 )
  {
    v6 = *a2;
    v7 = a1 + 13;
    do
    {
      while ( 1 )
      {
        v8 = v2[2];
        v9 = v2[3];
        if ( v2[4] >= v6 )
          break;
        v2 = (_QWORD *)v2[3];
        if ( !v9 )
          goto LABEL_6;
      }
      v7 = v2;
      v2 = (_QWORD *)v2[2];
    }
    while ( v8 );
LABEL_6:
    if ( v5 != v7 && v7[4] <= v6 )
      return 1;
  }
  v10 = (_QWORD *)a1[8];
  if ( !v10 )
    return 0;
  v11 = *a2;
  v12 = a1 + 7;
  do
  {
    if ( v10[4] < v11
      || v10[4] == v11
      && ((v13 = *((_DWORD *)a2 + 2), *((_DWORD *)v10 + 10) < v13)
       || *((_DWORD *)v10 + 10) == v13 && *((_BYTE *)v10 + 44) < *((_BYTE *)a2 + 12)) )
    {
      v10 = (_QWORD *)v10[3];
    }
    else
    {
      v12 = v10;
      v10 = (_QWORD *)v10[2];
    }
  }
  while ( v10 );
  if ( a1 + 7 == v12 )
    return 0;
  if ( v11 < v12[4] )
    return 0;
  if ( v11 == v12[4]
    && ((v14 = *((_DWORD *)v12 + 10), *((_DWORD *)a2 + 2) < v14)
     || *((_DWORD *)a2 + 2) == v14 && *((_BYTE *)a2 + 12) < *((_BYTE *)v12 + 44)) )
  {
    return 0;
  }
  else
  {
    return 1;
  }
}
