// Function: sub_26BDBC0
// Address: 0x26bdbc0
//
_QWORD *__fastcall sub_26BDBC0(__int64 a1, int a2)
{
  unsigned __int8 v4; // r8
  __int64 v5; // rbx
  __int64 v6; // rsi
  char v7; // al
  _BYTE *v8; // rdx
  unsigned __int8 v9; // al
  __int64 v10; // rsi
  unsigned __int8 v11; // al
  _BYTE **v12; // rdx
  __int64 *v13; // rdi
  __int64 v14; // rcx
  unsigned __int8 v15; // al
  __int64 v16; // r8
  __int64 *v17; // rdi
  __int64 v19; // rbx

  v4 = *(_BYTE *)(a1 - 16);
  v5 = a1 - 16;
  if ( (v4 & 2) != 0 )
  {
    v6 = **(_QWORD **)(a1 - 32);
    v7 = *(_BYTE *)v6;
    v8 = (_BYTE *)v6;
    if ( *(_BYTE *)v6 != 20 )
      goto LABEL_10;
  }
  else
  {
    v6 = *(_QWORD *)(v5 - 8LL * ((v4 >> 2) & 0xF));
    v7 = *(_BYTE *)v6;
    v8 = (_BYTE *)v6;
    if ( *(_BYTE *)v6 != 20 )
      goto LABEL_10;
  }
  do
  {
    if ( !*(_DWORD *)(v6 + 4) )
      break;
    v9 = *(_BYTE *)(v6 - 16);
    v10 = (v9 & 2) != 0 ? *(_QWORD *)(v6 - 32) : v6 - 16 - 8LL * ((v9 >> 2) & 0xF);
    v6 = *(_QWORD *)(v10 + 8);
  }
  while ( *(_BYTE *)v6 == 20 );
  if ( (v4 & 2) != 0 )
    v8 = **(_BYTE ***)(a1 - 32);
  else
    v8 = *(_BYTE **)(v5 - 8LL * ((v4 >> 2) & 0xF));
  v7 = *v8;
LABEL_10:
  if ( v7 != 16 )
  {
    v11 = *(v8 - 16);
    if ( (v11 & 2) != 0 )
      v12 = (_BYTE **)*((_QWORD *)v8 - 4);
    else
      v12 = (_BYTE **)&v8[-8 * ((v11 >> 2) & 0xF) - 16];
    v8 = *v12;
  }
  v13 = (__int64 *)(*(_QWORD *)(a1 + 8) & 0xFFFFFFFFFFFFFFF8LL);
  if ( (*(_QWORD *)(a1 + 8) & 4) != 0 )
    v13 = (__int64 *)*v13;
  v14 = sub_B09060(v13, v6, (__int64)v8, a2, 0, 1);
  v15 = *(_BYTE *)(a1 - 16);
  if ( (v15 & 2) != 0 )
  {
    v16 = 0;
    if ( *(_DWORD *)(a1 - 24) != 2 )
      goto LABEL_18;
    v19 = *(_QWORD *)(a1 - 32);
    goto LABEL_27;
  }
  v16 = 0;
  if ( ((*(_WORD *)(a1 - 16) >> 6) & 0xF) == 2 )
  {
    v19 = v5 - 8LL * ((v15 >> 2) & 0xF);
LABEL_27:
    v16 = *(_QWORD *)(v19 + 8);
  }
LABEL_18:
  v17 = (__int64 *)(*(_QWORD *)(a1 + 8) & 0xFFFFFFFFFFFFFFF8LL);
  if ( (*(_QWORD *)(a1 + 8) & 4) != 0 )
    v17 = (__int64 *)*v17;
  return sub_B01860(v17, *(_DWORD *)(a1 + 4), *(unsigned __int16 *)(a1 + 2), v14, v16, 0, 0, 1);
}
