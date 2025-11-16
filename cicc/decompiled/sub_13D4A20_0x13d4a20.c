// Function: sub_13D4A20
// Address: 0x13d4a20
//
__int64 __fastcall sub_13D4A20(__int64 a1, __int64 a2, char a3)
{
  __int16 v4; // r14
  __int16 v6; // r13
  int v8; // r13d
  __int64 v9; // r13
  __int64 v10; // r14
  char v11; // al
  __int64 v12; // rsi
  char v13; // al
  __int64 v15; // rdi
  bool v16; // al
  __int64 v17; // rdi
  bool v18; // al
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rax
  unsigned int v26; // ecx
  __int64 v27; // rax
  char v28; // si
  unsigned int v29; // ecx
  int v30; // eax
  bool v31; // al
  unsigned int v32; // ecx
  __int64 v33; // rax
  char v34; // si
  unsigned int v35; // ecx
  int v36; // eax
  bool v37; // al
  int v38; // [rsp+Ch] [rbp-54h]
  int v39; // [rsp+Ch] [rbp-54h]
  int v40; // [rsp+10h] [rbp-50h]
  int v41; // [rsp+10h] [rbp-50h]
  unsigned int v42; // [rsp+14h] [rbp-4Ch]
  unsigned int v43; // [rsp+14h] [rbp-4Ch]
  __int64 v44; // [rsp+18h] [rbp-48h]
  __int64 v45; // [rsp+18h] [rbp-48h]
  int v46; // [rsp+18h] [rbp-48h]
  int v47; // [rsp+18h] [rbp-48h]
  int v48; // [rsp+18h] [rbp-48h]
  int v49; // [rsp+18h] [rbp-48h]
  __int64 v50; // [rsp+18h] [rbp-48h]
  __int64 v51; // [rsp+18h] [rbp-48h]
  _QWORD v52[8]; // [rsp+20h] [rbp-40h] BYREF

  if ( *(_BYTE *)(*(_QWORD *)(a1 - 24) + 16LL) > 0x10u )
    return 0;
  v4 = *(_WORD *)(a1 + 18);
  v44 = *(_QWORD *)(a1 - 24);
  v6 = *(_WORD *)(a2 + 18);
  if ( !(unsigned __int8)sub_1593BB0(v44) )
  {
    if ( *(_BYTE *)(v44 + 16) == 13 )
    {
      if ( *(_DWORD *)(v44 + 32) <= 0x40u )
      {
        v18 = *(_QWORD *)(v44 + 24) == 0;
      }
      else
      {
        v17 = v44 + 24;
        v47 = *(_DWORD *)(v44 + 32);
        v18 = v47 == (unsigned int)sub_16A57B0(v17);
      }
    }
    else
    {
      if ( *(_BYTE *)(*(_QWORD *)v44 + 8LL) != 16 )
        return 0;
      v21 = sub_15A1020(v44);
      v22 = v44;
      if ( !v21 || *(_BYTE *)(v21 + 16) != 13 )
      {
        v40 = *(_QWORD *)(*(_QWORD *)v44 + 32LL);
        if ( v40 )
        {
          v26 = 0;
          while ( 1 )
          {
            v42 = v26;
            v50 = v22;
            v27 = sub_15A0A60(v22, v26);
            if ( !v27 )
              return 0;
            v28 = *(_BYTE *)(v27 + 16);
            v22 = v50;
            v29 = v42;
            if ( v28 != 9 )
            {
              if ( v28 != 13 )
                return 0;
              if ( *(_DWORD *)(v27 + 32) <= 0x40u )
              {
                v31 = *(_QWORD *)(v27 + 24) == 0;
              }
              else
              {
                v38 = *(_DWORD *)(v27 + 32);
                v30 = sub_16A57B0(v27 + 24);
                v22 = v50;
                v29 = v42;
                v31 = v38 == v30;
              }
              if ( !v31 )
                return 0;
            }
            v26 = v29 + 1;
            if ( v40 == v26 )
              goto LABEL_3;
          }
        }
        goto LABEL_3;
      }
      if ( *(_DWORD *)(v21 + 32) <= 0x40u )
      {
        v18 = *(_QWORD *)(v21 + 24) == 0;
      }
      else
      {
        v49 = *(_DWORD *)(v21 + 32);
        v18 = v49 == (unsigned int)sub_16A57B0(v21 + 24);
      }
    }
    if ( !v18 )
      return 0;
  }
LABEL_3:
  if ( *(_BYTE *)(*(_QWORD *)(a2 - 24) + 16LL) > 0x10u )
    return 0;
  v45 = *(_QWORD *)(a2 - 24);
  if ( !(unsigned __int8)sub_1593BB0(v45) )
  {
    if ( *(_BYTE *)(v45 + 16) == 13 )
    {
      if ( *(_DWORD *)(v45 + 32) <= 0x40u )
      {
        v16 = *(_QWORD *)(v45 + 24) == 0;
      }
      else
      {
        v15 = v45 + 24;
        v46 = *(_DWORD *)(v45 + 32);
        v16 = v46 == (unsigned int)sub_16A57B0(v15);
      }
    }
    else
    {
      if ( *(_BYTE *)(*(_QWORD *)v45 + 8LL) != 16 )
        return 0;
      v19 = sub_15A1020(v45);
      v20 = v45;
      if ( !v19 || *(_BYTE *)(v19 + 16) != 13 )
      {
        v41 = *(_QWORD *)(*(_QWORD *)v45 + 32LL);
        if ( v41 )
        {
          v32 = 0;
          while ( 1 )
          {
            v43 = v32;
            v51 = v20;
            v33 = sub_15A0A60(v20, v32);
            if ( !v33 )
              return 0;
            v34 = *(_BYTE *)(v33 + 16);
            v20 = v51;
            v35 = v43;
            if ( v34 != 9 )
            {
              if ( v34 != 13 )
                return 0;
              if ( *(_DWORD *)(v33 + 32) <= 0x40u )
              {
                v37 = *(_QWORD *)(v33 + 24) == 0;
              }
              else
              {
                v39 = *(_DWORD *)(v33 + 32);
                v36 = sub_16A57B0(v33 + 24);
                v20 = v51;
                v35 = v43;
                v37 = v39 == v36;
              }
              if ( !v37 )
                return 0;
            }
            v32 = v35 + 1;
            if ( v41 == v32 )
              goto LABEL_5;
          }
        }
        goto LABEL_5;
      }
      if ( *(_DWORD *)(v19 + 32) <= 0x40u )
      {
        v16 = *(_QWORD *)(v19 + 24) == 0;
      }
      else
      {
        v48 = *(_DWORD *)(v19 + 32);
        v16 = v48 == (unsigned int)sub_16A57B0(v19 + 24);
      }
    }
    if ( !v16 )
      return 0;
  }
LABEL_5:
  v8 = v6 & 0x7FFF;
  if ( (v4 & 0x7FFF) != v8 || v8 != 33 && a3 || a3 != 1 && v8 != 32 )
    return 0;
  v9 = *(_QWORD *)(a2 - 48);
  v10 = *(_QWORD *)(a1 - 48);
  v11 = *(_BYTE *)(v9 + 16);
  if ( v11 == 50 )
  {
    if ( v10 == *(_QWORD *)(v9 - 48) || v10 == *(_QWORD *)(v9 - 24) )
      return a2;
  }
  else if ( v11 == 5
         && *(_WORD *)(v9 + 18) == 26
         && (v10 == *(_QWORD *)(v9 - 24LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF))
          || v10 == *(_QWORD *)(v9 + 24 * (1LL - (*(_DWORD *)(v9 + 20) & 0xFFFFFFF)))) )
  {
    return a2;
  }
  v12 = *(_QWORD *)(a2 - 48);
  v52[0] = *(_QWORD *)(a1 - 48);
  if ( !(unsigned __int8)sub_13D4880(v52, v12) )
  {
    v13 = *(_BYTE *)(v10 + 16);
    if ( v13 == 50 )
    {
      v25 = *(_QWORD *)(v10 - 48);
      if ( v9 == v25 && v25 )
        return a1;
      v24 = *(_QWORD *)(v10 - 24);
      if ( v9 != v24 )
        goto LABEL_17;
    }
    else
    {
      if ( v13 != 5 || *(_WORD *)(v10 + 18) != 26 )
        goto LABEL_17;
      v23 = *(_QWORD *)(v10 - 24LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF));
      if ( v23 && v9 == v23 )
        return a1;
      v24 = *(_QWORD *)(v10 + 24 * (1LL - (*(_DWORD *)(v10 + 20) & 0xFFFFFFF)));
      if ( v9 != v24 )
        goto LABEL_17;
    }
    if ( v24 )
      return a1;
LABEL_17:
    v52[0] = v9;
    if ( (unsigned __int8)sub_13D4880(v52, v10) )
      return a1;
    return 0;
  }
  return a2;
}
