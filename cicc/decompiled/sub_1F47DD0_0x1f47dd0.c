// Function: sub_1F47DD0
// Address: 0x1f47dd0
//
bool __fastcall sub_1F47DD0(__int64 a1)
{
  unsigned __int64 v2; // rsi
  _QWORD *v3; // rax
  _DWORD *v4; // rdi
  __int64 v5; // rcx
  __int64 v6; // rdx
  __int64 v7; // rax
  _DWORD *v8; // r8
  _DWORD *v9; // rdi
  __int64 v10; // rcx
  __int64 v11; // rdx

  v2 = sub_16D5D50();
  v3 = *(_QWORD **)&dword_4FA0208[2];
  if ( !*(_QWORD *)&dword_4FA0208[2] )
    return ((*(_BYTE *)(*(_QWORD *)(a1 + 208) + 800LL) >> 2) ^ 1) & 1;
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
    return ((*(_BYTE *)(*(_QWORD *)(a1 + 208) + 800LL) >> 2) ^ 1) & 1;
  if ( v2 < *((_QWORD *)v4 + 4) )
    return ((*(_BYTE *)(*(_QWORD *)(a1 + 208) + 800LL) >> 2) ^ 1) & 1;
  v7 = *((_QWORD *)v4 + 7);
  v8 = v4 + 12;
  if ( !v7 )
    return ((*(_BYTE *)(*(_QWORD *)(a1 + 208) + 800LL) >> 2) ^ 1) & 1;
  v9 = v4 + 12;
  do
  {
    while ( 1 )
    {
      v10 = *(_QWORD *)(v7 + 16);
      v11 = *(_QWORD *)(v7 + 24);
      if ( *(_DWORD *)(v7 + 32) >= dword_4FCC2E8 )
        break;
      v7 = *(_QWORD *)(v7 + 24);
      if ( !v11 )
        goto LABEL_13;
    }
    v9 = (_DWORD *)v7;
    v7 = *(_QWORD *)(v7 + 16);
  }
  while ( v10 );
LABEL_13:
  if ( v8 == v9 || dword_4FCC2E8 < v9[8] || (int)v9[9] <= 0 )
    return ((*(_BYTE *)(*(_QWORD *)(a1 + 208) + 800LL) >> 2) ^ 1) & 1;
  else
    return dword_4FCC380 == 1;
}
