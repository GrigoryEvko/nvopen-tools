// Function: sub_17C5A30
// Address: 0x17c5a30
//
__int64 __fastcall sub_17C5A30(__int64 a1)
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
  __int64 result; // rax

  v2 = sub_16D5D50();
  v3 = *(_QWORD **)&dword_4FA0208[2];
  if ( !*(_QWORD *)&dword_4FA0208[2] )
    return *(unsigned __int8 *)(a1 + 1);
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
    return *(unsigned __int8 *)(a1 + 1);
  if ( v2 < *((_QWORD *)v4 + 4) )
    return *(unsigned __int8 *)(a1 + 1);
  v7 = *((_QWORD *)v4 + 7);
  v8 = v4 + 12;
  if ( !v7 )
    return *(unsigned __int8 *)(a1 + 1);
  v9 = v4 + 12;
  do
  {
    while ( 1 )
    {
      v10 = *(_QWORD *)(v7 + 16);
      v11 = *(_QWORD *)(v7 + 24);
      if ( *(_DWORD *)(v7 + 32) >= dword_4FA3728 )
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
  if ( v8 == v9 )
    return *(unsigned __int8 *)(a1 + 1);
  if ( dword_4FA3728 < v9[8] )
    return *(unsigned __int8 *)(a1 + 1);
  result = (unsigned __int8)byte_4FA37C0;
  if ( (int)v9[9] <= 0 )
    return *(unsigned __int8 *)(a1 + 1);
  return result;
}
