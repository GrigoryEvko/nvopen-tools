// Function: sub_1F40C40
// Address: 0x1f40c40
//
unsigned __int64 __fastcall sub_1F40C40(__int64 a1, char a2)
{
  unsigned __int64 v4; // rsi
  unsigned __int64 result; // rax
  _DWORD *v6; // rdi
  __int64 v7; // rcx
  __int64 v8; // rdx
  _DWORD *v9; // r8
  _DWORD *v10; // rdi
  __int64 v11; // rcx
  __int64 v12; // rdx

  v4 = sub_16D5D50();
  result = *(_QWORD *)&dword_4FA0208[2];
  if ( !*(_QWORD *)&dword_4FA0208[2] )
    goto LABEL_15;
  v6 = dword_4FA0208;
  do
  {
    while ( 1 )
    {
      v7 = *(_QWORD *)(result + 16);
      v8 = *(_QWORD *)(result + 24);
      if ( v4 <= *(_QWORD *)(result + 32) )
        break;
      result = *(_QWORD *)(result + 24);
      if ( !v8 )
        goto LABEL_6;
    }
    v6 = (_DWORD *)result;
    result = *(_QWORD *)(result + 16);
  }
  while ( v7 );
LABEL_6:
  result = (unsigned __int64)dword_4FA0208;
  if ( v6 == dword_4FA0208 )
    goto LABEL_15;
  if ( v4 < *((_QWORD *)v6 + 4) )
    goto LABEL_15;
  result = *((_QWORD *)v6 + 7);
  v9 = v6 + 12;
  if ( !result )
    goto LABEL_15;
  v10 = v6 + 12;
  do
  {
    while ( 1 )
    {
      v11 = *(_QWORD *)(result + 16);
      v12 = *(_QWORD *)(result + 24);
      if ( *(_DWORD *)(result + 32) >= dword_4FCB648 )
        break;
      result = *(_QWORD *)(result + 24);
      if ( !v12 )
        goto LABEL_13;
    }
    v10 = (_DWORD *)result;
    result = *(_QWORD *)(result + 16);
  }
  while ( v11 );
LABEL_13:
  if ( v9 == v10 || dword_4FCB648 < v10[8] || (result = (unsigned int)v10[9], !(_DWORD)result) )
LABEL_15:
    *(_BYTE *)(a1 + 56) = a2;
  return result;
}
