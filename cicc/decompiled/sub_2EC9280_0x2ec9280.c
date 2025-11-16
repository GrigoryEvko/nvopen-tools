// Function: sub_2EC9280
// Address: 0x2ec9280
//
__int64 __fastcall sub_2EC9280(__int64 a1, __int64 a2, _DWORD *a3)
{
  __int64 v5; // r14
  __int64 v6; // r15
  unsigned int v7; // r14d
  unsigned int v8; // eax
  __int64 v9; // rbx
  int v10; // r14d
  __int64 v11; // rbx
  __int64 v12; // rbx
  int v13; // r14d
  __int64 v14; // rbx
  int v15; // edi
  unsigned __int8 v16; // r8
  __int64 v18; // r15
  unsigned int v19; // r14d
  unsigned int v20; // eax
  __int64 v21; // rbx
  int v22; // r14d
  __int64 v23; // rbx
  __int64 v24; // rbx
  __int64 v25; // rbx

  v5 = *(_QWORD *)(a2 + 16);
  if ( a3[6] == 1 )
  {
    if ( (*(_BYTE *)(v5 + 254) & 1) == 0 )
      sub_2F8F5D0(*(_QWORD *)(a2 + 16));
    v18 = *(_QWORD *)(a1 + 16);
    v19 = *(_DWORD *)(v5 + 240);
    if ( (*(_BYTE *)(v18 + 254) & 1) == 0 )
      sub_2F8F5D0(*(_QWORD *)(a1 + 16));
    v20 = a3[41];
    if ( *(_DWORD *)(v18 + 240) >= v19 )
      v19 = *(_DWORD *)(v18 + 240);
    if ( a3[44] >= v20 )
      v20 = a3[44];
    if ( v19 <= v20 )
      goto LABEL_37;
    v21 = *(_QWORD *)(a2 + 16);
    if ( (*(_BYTE *)(v21 + 254) & 1) == 0 )
      sub_2F8F5D0(*(_QWORD *)(a2 + 16));
    v22 = *(_DWORD *)(v21 + 240);
    v23 = *(_QWORD *)(a1 + 16);
    if ( (*(_BYTE *)(v23 + 254) & 1) == 0 )
      sub_2F8F5D0(*(_QWORD *)(a1 + 16));
    if ( !(unsigned __int8)sub_2EC9220(*(_DWORD *)(v23 + 240), v22, a1, a2, 0xDu) )
    {
LABEL_37:
      v24 = *(_QWORD *)(a2 + 16);
      if ( (*(_BYTE *)(v24 + 254) & 2) == 0 )
        sub_2F8F770(*(_QWORD *)(a2 + 16));
      v13 = *(_DWORD *)(v24 + 244);
      v25 = *(_QWORD *)(a1 + 16);
      if ( (*(_BYTE *)(v25 + 254) & 2) == 0 )
        sub_2F8F770(*(_QWORD *)(a1 + 16));
      v15 = *(_DWORD *)(v25 + 244);
      v16 = 14;
      return sub_2EC9250(v15, v13, a1, a2, v16);
    }
  }
  else
  {
    if ( (*(_BYTE *)(v5 + 254) & 2) == 0 )
      sub_2F8F770(*(_QWORD *)(a2 + 16));
    v6 = *(_QWORD *)(a1 + 16);
    v7 = *(_DWORD *)(v5 + 244);
    if ( (*(_BYTE *)(v6 + 254) & 2) == 0 )
      sub_2F8F770(*(_QWORD *)(a1 + 16));
    v8 = a3[41];
    if ( *(_DWORD *)(v6 + 244) >= v7 )
      v7 = *(_DWORD *)(v6 + 244);
    if ( a3[44] >= v8 )
      v8 = a3[44];
    if ( v7 <= v8 )
      goto LABEL_16;
    v9 = *(_QWORD *)(a2 + 16);
    if ( (*(_BYTE *)(v9 + 254) & 2) == 0 )
      sub_2F8F770(*(_QWORD *)(a2 + 16));
    v10 = *(_DWORD *)(v9 + 244);
    v11 = *(_QWORD *)(a1 + 16);
    if ( (*(_BYTE *)(v11 + 254) & 2) == 0 )
      sub_2F8F770(*(_QWORD *)(a1 + 16));
    if ( !(unsigned __int8)sub_2EC9220(*(_DWORD *)(v11 + 244), v10, a1, a2, 0xBu) )
    {
LABEL_16:
      v12 = *(_QWORD *)(a2 + 16);
      if ( (*(_BYTE *)(v12 + 254) & 1) == 0 )
        sub_2F8F5D0(*(_QWORD *)(a2 + 16));
      v13 = *(_DWORD *)(v12 + 240);
      v14 = *(_QWORD *)(a1 + 16);
      if ( (*(_BYTE *)(v14 + 254) & 1) == 0 )
        sub_2F8F5D0(*(_QWORD *)(a1 + 16));
      v15 = *(_DWORD *)(v14 + 240);
      v16 = 12;
      return sub_2EC9250(v15, v13, a1, a2, v16);
    }
  }
  return 1;
}
