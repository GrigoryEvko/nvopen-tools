// Function: sub_385D890
// Address: 0x385d890
//
__int64 sub_385D890()
{
  unsigned __int64 v0; // rsi
  _QWORD *v1; // rax
  _DWORD *v2; // rdi
  __int64 v3; // rcx
  __int64 v4; // rdx
  __int64 v5; // rax
  _DWORD *v6; // r9
  _DWORD *v7; // rdi
  __int64 v8; // rcx
  __int64 v9; // rdx
  unsigned int v10; // r8d

  v0 = sub_16D5D50();
  v1 = *(_QWORD **)&dword_4FA0208[2];
  if ( !*(_QWORD *)&dword_4FA0208[2] )
    return 0;
  v2 = dword_4FA0208;
  do
  {
    while ( 1 )
    {
      v3 = v1[2];
      v4 = v1[3];
      if ( v0 <= v1[4] )
        break;
      v1 = (_QWORD *)v1[3];
      if ( !v4 )
        goto LABEL_6;
    }
    v2 = v1;
    v1 = (_QWORD *)v1[2];
  }
  while ( v3 );
LABEL_6:
  if ( v2 == dword_4FA0208 )
    return 0;
  if ( v0 < *((_QWORD *)v2 + 4) )
    return 0;
  v5 = *((_QWORD *)v2 + 7);
  v6 = v2 + 12;
  if ( !v5 )
    return 0;
  v7 = v2 + 12;
  do
  {
    while ( 1 )
    {
      v8 = *(_QWORD *)(v5 + 16);
      v9 = *(_QWORD *)(v5 + 24);
      if ( *(_DWORD *)(v5 + 32) >= dword_5052408 )
        break;
      v5 = *(_QWORD *)(v5 + 24);
      if ( !v9 )
        goto LABEL_13;
    }
    v7 = (_DWORD *)v5;
    v5 = *(_QWORD *)(v5 + 16);
  }
  while ( v8 );
LABEL_13:
  v10 = 0;
  if ( v6 == v7 || dword_5052408 < v7[8] )
    return 0;
  LOBYTE(v10) = v7[9] > 0;
  return v10;
}
