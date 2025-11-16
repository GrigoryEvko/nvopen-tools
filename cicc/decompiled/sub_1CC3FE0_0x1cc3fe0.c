// Function: sub_1CC3FE0
// Address: 0x1cc3fe0
//
__int64 __fastcall sub_1CC3FE0(
        __int64 a1,
        int a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8)
{
  __int64 result; // rax
  unsigned int v10; // r12d
  unsigned __int64 v11; // rsi
  _QWORD *v12; // rax
  _DWORD *v13; // rdi
  __int64 v14; // rcx
  __int64 v15; // rdx
  __int64 v16; // rax
  _DWORD *v17; // r9
  int v18; // esi
  _DWORD *v19; // rdi
  __int64 v20; // rcx
  __int64 v21; // rdx
  _DWORD v22[9]; // [rsp+14h] [rbp-24h] BYREF

  v22[0] = 0;
  LODWORD(result) = sub_16B3650(a1 + 208, a1, a4, a5, a7, a8, v22);
  v10 = result;
  if ( (_BYTE)result )
    return (unsigned int)result;
  *(_QWORD *)(a1 + 168) -= 4LL;
  *(_QWORD *)(a1 + 192) -= 4LL;
  *(_DWORD *)(a1 + 16) = a2;
  v11 = sub_16D5D50();
  v12 = *(_QWORD **)&dword_4FA0208[2];
  if ( !*(_QWORD *)&dword_4FA0208[2] )
    goto LABEL_17;
  v13 = dword_4FA0208;
  do
  {
    while ( 1 )
    {
      v14 = v12[2];
      v15 = v12[3];
      if ( v11 <= v12[4] )
        break;
      v12 = (_QWORD *)v12[3];
      if ( !v15 )
        goto LABEL_8;
    }
    v13 = v12;
    v12 = (_QWORD *)v12[2];
  }
  while ( v14 );
LABEL_8:
  if ( v13 == dword_4FA0208 )
    goto LABEL_17;
  if ( v11 < *((_QWORD *)v13 + 4) )
    goto LABEL_17;
  v16 = *((_QWORD *)v13 + 7);
  v17 = v13 + 12;
  if ( !v16 )
    goto LABEL_17;
  v18 = *(_DWORD *)(a1 + 8);
  v19 = v13 + 12;
  do
  {
    while ( 1 )
    {
      v20 = *(_QWORD *)(v16 + 16);
      v21 = *(_QWORD *)(v16 + 24);
      if ( *(_DWORD *)(v16 + 32) >= v18 )
        break;
      v16 = *(_QWORD *)(v16 + 24);
      if ( !v21 )
        goto LABEL_15;
    }
    v19 = (_DWORD *)v16;
    v16 = *(_QWORD *)(v16 + 16);
  }
  while ( v20 );
LABEL_15:
  if ( v17 != v19 && v18 >= v19[8] )
    sub_1CC3DA0((int *)(a1 + 8), v19[9] - 1);
  else
LABEL_17:
    sub_1CC3DA0((int *)(a1 + 8), 0xFFFFFFFF);
  return v10;
}
