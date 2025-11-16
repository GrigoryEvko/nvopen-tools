// Function: sub_1E73920
// Address: 0x1e73920
//
__int64 __fastcall sub_1E73920(__int64 a1, __int64 a2, _DWORD *a3)
{
  __int64 v5; // r13
  unsigned int v6; // eax
  unsigned int v7; // r15d
  __int64 v8; // rbx
  __int64 v9; // rbx
  int v10; // r13d
  int v11; // edi
  unsigned __int8 v12; // r8
  unsigned int v14; // eax
  __int64 v15; // rbx
  unsigned int v16; // eax
  unsigned int v17; // r15d
  __int64 v18; // rbx
  __int64 v19; // rbx
  unsigned int v20; // eax
  __int64 v21; // rbx

  v5 = *(_QWORD *)(a2 + 16);
  if ( a3[6] == 1 )
  {
    if ( (*(_BYTE *)(v5 + 236) & 1) != 0 )
    {
      v16 = a3[44];
      v17 = *(_DWORD *)(v5 + 240);
      if ( a3[41] >= v16 )
        v16 = a3[41];
      if ( v17 <= v16 )
        goto LABEL_30;
    }
    else
    {
      sub_1F01DD0(*(_QWORD *)(a2 + 16));
      v20 = a3[41];
      if ( a3[44] >= v20 )
        v20 = a3[44];
      if ( *(_DWORD *)(v5 + 240) <= v20 )
      {
LABEL_29:
        v5 = *(_QWORD *)(a2 + 16);
LABEL_30:
        if ( (*(_BYTE *)(v5 + 236) & 2) == 0 )
          sub_1F01F70(v5);
        v19 = *(_QWORD *)(a1 + 16);
        v10 = *(_DWORD *)(v5 + 244);
        if ( (*(_BYTE *)(v19 + 236) & 2) == 0 )
          sub_1F01F70(*(_QWORD *)(a1 + 16));
        v11 = *(_DWORD *)(v19 + 244);
        v12 = 14;
        return sub_1E738F0(v11, v10, a1, a2, v12);
      }
      v21 = *(_QWORD *)(a2 + 16);
      if ( (*(_BYTE *)(v21 + 236) & 1) == 0 )
        sub_1F01DD0(*(_QWORD *)(a2 + 16));
      v17 = *(_DWORD *)(v21 + 240);
    }
    v18 = *(_QWORD *)(a1 + 16);
    if ( (*(_BYTE *)(v18 + 236) & 1) == 0 )
      sub_1F01DD0(*(_QWORD *)(a1 + 16));
    if ( (unsigned __int8)sub_1E738C0(*(_DWORD *)(v18 + 240), v17, a1, a2, 0xDu) )
      return 1;
    goto LABEL_29;
  }
  if ( (*(_BYTE *)(v5 + 236) & 2) == 0 )
  {
    sub_1F01F70(*(_QWORD *)(a2 + 16));
    v14 = a3[41];
    if ( a3[44] >= v14 )
      v14 = a3[44];
    if ( v14 >= *(_DWORD *)(v5 + 244) )
      goto LABEL_8;
    v15 = *(_QWORD *)(a2 + 16);
    if ( (*(_BYTE *)(v15 + 236) & 2) == 0 )
      sub_1F01F70(*(_QWORD *)(a2 + 16));
    v7 = *(_DWORD *)(v15 + 244);
    v8 = *(_QWORD *)(a1 + 16);
    if ( (*(_BYTE *)(v8 + 236) & 2) != 0 )
      goto LABEL_7;
    goto LABEL_21;
  }
  v6 = a3[41];
  v7 = *(_DWORD *)(v5 + 244);
  if ( a3[44] >= v6 )
    v6 = a3[44];
  if ( v7 <= v6 )
    goto LABEL_9;
  v8 = *(_QWORD *)(a1 + 16);
  if ( (*(_BYTE *)(v8 + 236) & 2) == 0 )
LABEL_21:
    sub_1F01F70(v8);
LABEL_7:
  if ( !(unsigned __int8)sub_1E738C0(*(_DWORD *)(v8 + 244), v7, a1, a2, 0xBu) )
  {
LABEL_8:
    v5 = *(_QWORD *)(a2 + 16);
LABEL_9:
    if ( (*(_BYTE *)(v5 + 236) & 1) == 0 )
      sub_1F01DD0(v5);
    v9 = *(_QWORD *)(a1 + 16);
    v10 = *(_DWORD *)(v5 + 240);
    if ( (*(_BYTE *)(v9 + 236) & 1) == 0 )
      sub_1F01DD0(*(_QWORD *)(a1 + 16));
    v11 = *(_DWORD *)(v9 + 240);
    v12 = 12;
    return sub_1E738F0(v11, v10, a1, a2, v12);
  }
  return 1;
}
