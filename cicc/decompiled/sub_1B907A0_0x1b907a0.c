// Function: sub_1B907A0
// Address: 0x1b907a0
//
__int64 __fastcall sub_1B907A0(int *a1)
{
  unsigned __int64 v2; // rsi
  _QWORD *v3; // rax
  _DWORD *v4; // rdi
  __int64 v5; // rcx
  __int64 v6; // rdx
  __int64 v7; // rax
  _DWORD *v8; // r9
  int v9; // esi
  _DWORD *v10; // rdi
  __int64 v11; // rcx
  __int64 v12; // rdx

  v2 = sub_16D5D50();
  v3 = *(_QWORD **)&dword_4FA0208[2];
  if ( !*(_QWORD *)&dword_4FA0208[2] )
    return 0;
  v4 = dword_4FA0208;
  do
  {
    while ( 1 )
    {
      v5 = v3[2];
      v6 = v3[3];
      if ( v2 <= v3[4] )
        break;
      v3 = (_QWORD *)v3[3];
      if ( !v6 )
        goto LABEL_6;
    }
    v4 = v3;
    v3 = (_QWORD *)v3[2];
  }
  while ( v5 );
LABEL_6:
  if ( v4 == dword_4FA0208 )
    return 0;
  if ( v2 < *((_QWORD *)v4 + 4) )
    return 0;
  v7 = *((_QWORD *)v4 + 7);
  v8 = v4 + 12;
  if ( !v7 )
    return 0;
  v9 = *a1;
  v10 = v4 + 12;
  do
  {
    while ( 1 )
    {
      v11 = *(_QWORD *)(v7 + 16);
      v12 = *(_QWORD *)(v7 + 24);
      if ( *(_DWORD *)(v7 + 32) >= v9 )
        break;
      v7 = *(_QWORD *)(v7 + 24);
      if ( !v12 )
        goto LABEL_13;
    }
    v10 = (_DWORD *)v7;
    v7 = *(_QWORD *)(v7 + 16);
  }
  while ( v11 );
LABEL_13:
  if ( v8 == v10 || v9 < v10[8] )
    return 0;
  else
    return (unsigned int)v10[9];
}
