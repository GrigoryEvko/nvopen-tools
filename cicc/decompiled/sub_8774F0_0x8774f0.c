// Function: sub_8774F0
// Address: 0x8774f0
//
__int64 __fastcall sub_8774F0(__int64 a1, __int64 a2, FILE *a3, char a4, unsigned int a5, int a6, _DWORD *a7)
{
  char v7; // al
  char v8; // r10
  __int64 result; // rax
  __int64 v10; // rsi
  char v11; // al
  __int64 v12; // rax
  __int64 v13; // rax
  char v14; // al
  int v15; // r8d

  v7 = *(_BYTE *)(a1 + 80);
  if ( v7 == 16 )
  {
    a1 = **(_QWORD **)(a1 + 88);
    v7 = *(_BYTE *)(a1 + 80);
  }
  if ( v7 == 24 )
  {
    a1 = *(_QWORD *)(a1 + 88);
    v7 = *(_BYTE *)(a1 + 80);
  }
  if ( (unsigned __int8)(v7 - 10) <= 1u )
  {
    v13 = *(_QWORD *)(a1 + 88);
    if ( (*(_BYTE *)(v13 + 194) & 0x40) != 0 )
    {
      do
        v13 = *(_QWORD *)(v13 + 232);
      while ( (*(_BYTE *)(v13 + 194) & 0x40) != 0 );
      a1 = *(_QWORD *)v13;
    }
  }
  else if ( v7 == 20 )
  {
    v12 = *(_QWORD *)(*(_QWORD *)(a1 + 88) + 176LL);
    if ( (*(_BYTE *)(v12 + 194) & 0x40) != 0 )
    {
      do
        v12 = *(_QWORD *)(v12 + 232);
      while ( (*(_BYTE *)(v12 + 194) & 0x40) != 0 );
      a1 = **(_QWORD **)(v12 + 248);
    }
  }
  if ( a5 )
    goto LABEL_20;
  if ( a2 )
  {
    v8 = 7;
    a5 = 410;
    if ( !a7 )
      return sub_686750(7u, 0x19Au, a3, a1, a2);
    goto LABEL_22;
  }
  v14 = *(_BYTE *)(a1 + 80);
  if ( (unsigned __int8)(v14 - 10) <= 1u )
  {
    v10 = *(_QWORD *)(a1 + 88);
  }
  else
  {
    if ( v14 != 17 )
    {
      if ( v14 == 3 || (a5 = 265, a4 = 7, dword_4F077C4 == 2) && (unsigned __int8)(v14 - 4) <= 2u )
      {
        v15 = -(qword_4D0495C == 0);
        LOBYTE(v15) = v15 & 0x27;
        a5 = v15 + 482;
        a4 = qword_4D0495C == 0 ? 7 : 5;
      }
      goto LABEL_16;
    }
    v10 = *(_QWORD *)(*(_QWORD *)(a1 + 88) + 88LL);
  }
  v11 = *(_BYTE *)(v10 + 174);
  if ( (unsigned __int8)(v11 - 1) <= 2u )
  {
    a5 = 330;
    a4 = 7;
  }
  else
  {
    a5 = 265;
    a4 = 7;
    if ( v11 == 5 && *(_BYTE *)(v10 + 176) == 15 )
      a5 = 330;
  }
LABEL_16:
  if ( !dword_4F077BC || (_DWORD)qword_4F077B4 || (unsigned __int64)(qword_4F077A8 - 30400LL) > 0x2903 || (v8 = 5, !a6) )
LABEL_20:
    v8 = a4;
  if ( a7 )
  {
LABEL_22:
    result = sub_67D3C0((int *)a5, v8, a3);
    *a7 = result;
    return result;
  }
  return sub_6853B0(v8, a5, a3, a1);
}
