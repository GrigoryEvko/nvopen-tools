// Function: sub_8A6950
// Address: 0x8a6950
//
__int64 __fastcall sub_8A6950(unsigned __int64 a1, unsigned __int64 a2)
{
  char v3; // al
  __int64 v4; // r15
  __int64 *v5; // rcx
  __int64 v6; // rsi
  _BOOL4 v7; // eax
  __int64 v8; // r9
  _BOOL4 v9; // ebx
  char v10; // al
  __int64 *v11; // rcx
  __int64 v12; // rsi
  _BOOL4 v13; // eax
  bool v14; // cl
  _BOOL4 v15; // r14d
  int v16; // r10d
  unsigned int v17; // r8d
  char v19; // dl
  unsigned int v20; // eax
  unsigned int v21; // eax
  bool v22; // [rsp+3h] [rbp-4Dh]
  int v23; // [rsp+4h] [rbp-4Ch]
  __int64 v24; // [rsp+8h] [rbp-48h]
  unsigned int v25; // [rsp+8h] [rbp-48h]
  __int64 *v26; // [rsp+10h] [rbp-40h] BYREF
  __int64 *v27; // [rsp+18h] [rbp-38h] BYREF

  v3 = *(_BYTE *)(a1 + 80);
  v26 = 0;
  v27 = 0;
  v4 = *(_QWORD *)(a1 + 88);
  if ( v3 == 19 )
  {
    v6 = *(_QWORD *)(v4 + 176);
  }
  else
  {
    v5 = *(__int64 **)(a1 + 88);
    if ( ((v3 - 7) & 0xFD) != 0 )
    {
      if ( v3 != 21 )
        BUG();
      v5 = *(__int64 **)(v4 + 192);
    }
    v6 = *v5;
  }
  v7 = sub_8A64A0(a2, v6, 1, &v26);
  v8 = *(_QWORD *)(a2 + 88);
  v9 = v7;
  v10 = *(_BYTE *)(a2 + 80);
  if ( v10 == 19 )
  {
    v12 = *(_QWORD *)(v8 + 176);
  }
  else
  {
    v11 = *(__int64 **)(a2 + 88);
    if ( ((v10 - 7) & 0xFD) != 0 )
    {
      if ( v10 != 21 )
        BUG();
      v11 = *(__int64 **)(v8 + 192);
    }
    v12 = *v11;
  }
  v24 = *(_QWORD *)(a2 + 88);
  v13 = sub_8A64A0(a1, v12, 1, &v27);
  v14 = v9;
  v15 = v13;
  if ( v13 )
  {
    v16 = 1;
    v17 = -1;
    if ( !v9 )
      goto LABEL_9;
LABEL_15:
    v19 = *(_BYTE *)(v24 + 160) & 0x10;
    if ( (*(_BYTE *)(v4 + 160) & 0x10) != 0 )
    {
      if ( v19 )
      {
        v20 = sub_88D570(v26, v27);
        v14 = v9;
        v17 = v20;
        goto LABEL_9;
      }
    }
    else
    {
      v16 = 0;
      v17 = 1;
      if ( v19 )
        goto LABEL_9;
    }
    v16 = 0;
    v17 = -((*(_BYTE *)(v4 + 160) & 0x10) != 0);
    goto LABEL_9;
  }
  v16 = 1;
  v17 = 1;
  if ( !v9 )
    goto LABEL_15;
LABEL_9:
  if ( dword_4D04494 )
  {
    v25 = v17;
    v23 = v16;
    v22 = v14;
    v21 = sub_6F3270(a1, a2, 0);
    v17 = v25;
    if ( v21 )
    {
      if ( v25 )
      {
        if ( v25 != v21 && !v23 )
          return 0;
      }
      else if ( v15 && v22 )
      {
        return v21;
      }
    }
  }
  return v17;
}
