// Function: sub_1F45FF0
// Address: 0x1f45ff0
//
__int64 __fastcall sub_1F45FF0(__int64 a1, _BYTE *a2, __int64 a3)
{
  _QWORD *v5; // rdx
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rax
  unsigned __int64 v9; // rsi
  _QWORD *v10; // rax
  _DWORD *v11; // rdi
  __int64 v12; // rcx
  __int64 v13; // rdx
  __int64 v14; // rax
  _DWORD *v15; // r8
  _DWORD *v16; // rdi
  __int64 v17; // rcx
  __int64 v18; // rdx
  char v19; // al
  __int64 (*v21)(); // rdx

  *(_QWORD *)(a1 + 16) = &unk_4FCBA30;
  *(_QWORD *)(a1 + 80) = a1 + 64;
  *(_QWORD *)(a1 + 88) = a1 + 64;
  *(_QWORD *)(a1 + 128) = a1 + 112;
  *(_QWORD *)(a1 + 136) = a1 + 112;
  *(_QWORD *)(a1 + 160) = a3;
  *(_WORD *)(a1 + 200) = 1;
  *(_QWORD *)a1 = &unk_49FF4C8;
  *(_QWORD *)(a1 + 208) = a2;
  *(_QWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 24) = 5;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_DWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  *(_DWORD *)(a1 + 112) = 0;
  *(_QWORD *)(a1 + 120) = 0;
  *(_QWORD *)(a1 + 144) = 0;
  *(_BYTE *)(a1 + 152) = 0;
  *(_QWORD *)(a1 + 168) = 0;
  *(_QWORD *)(a1 + 176) = 0;
  *(_QWORD *)(a1 + 184) = 0;
  *(_QWORD *)(a1 + 192) = 0;
  *(_BYTE *)(a1 + 202) = 0;
  *(_QWORD *)(a1 + 216) = 0;
  *(_DWORD *)(a1 + 224) = 0x10000;
  v5 = (_QWORD *)sub_22077B0(176);
  if ( v5 )
  {
    memset(v5, 0, 0xB0u);
    v5[4] = v5 + 6;
    v5[5] = 0x400000000LL;
  }
  *(_QWORD *)(a1 + 216) = v5;
  v6 = sub_163A1D0();
  sub_1D5A070(v6);
  v7 = sub_163A1D0();
  sub_1361770(v7);
  v8 = sub_163A1D0();
  sub_134D8E0(v8);
  if ( !qword_4FCC468 )
    a2[792] |= 1u;
  v9 = sub_16D5D50();
  v10 = *(_QWORD **)&dword_4FA0208[2];
  if ( !*(_QWORD *)&dword_4FA0208[2] )
    goto LABEL_24;
  v11 = dword_4FA0208;
  do
  {
    while ( 1 )
    {
      v12 = v10[2];
      v13 = v10[3];
      if ( v9 <= v10[4] )
        break;
      v10 = (_QWORD *)v10[3];
      if ( !v13 )
        goto LABEL_10;
    }
    v11 = v10;
    v10 = (_QWORD *)v10[2];
  }
  while ( v12 );
LABEL_10:
  if ( v11 == dword_4FA0208 )
    goto LABEL_24;
  if ( v9 < *((_QWORD *)v11 + 4) )
    goto LABEL_24;
  v14 = *((_QWORD *)v11 + 7);
  v15 = v11 + 12;
  if ( !v14 )
    goto LABEL_24;
  v16 = v11 + 12;
  v9 = LODWORD(qword_4FCDEE0[1]);
  do
  {
    while ( 1 )
    {
      v17 = *(_QWORD *)(v14 + 16);
      v18 = *(_QWORD *)(v14 + 24);
      if ( *(_DWORD *)(v14 + 32) >= (int)v9 )
        break;
      v14 = *(_QWORD *)(v14 + 24);
      if ( !v18 )
        goto LABEL_17;
    }
    v16 = (_DWORD *)v14;
    v14 = *(_QWORD *)(v14 + 16);
  }
  while ( v17 );
LABEL_17:
  if ( v15 != v16 && (int)v9 >= v16[8] && v16[9] )
  {
    v19 = qword_4FCDEE0[20];
  }
  else
  {
LABEL_24:
    v21 = *(__int64 (**)())(*(_QWORD *)a2 + 88LL);
    v19 = 0;
    if ( v21 != sub_16FF7C0 )
      v19 = ((__int64 (__fastcall *)(_BYTE *, unsigned __int64))v21)(a2, v9);
  }
  a2[809] = v19 & 1 | a2[809] & 0xFE;
  if ( (a2[809] & 1) != 0 )
    sub_1F45FE0(a1, (_BYTE *)(a1 + 227), 1);
  return sub_1F45B20(a1);
}
