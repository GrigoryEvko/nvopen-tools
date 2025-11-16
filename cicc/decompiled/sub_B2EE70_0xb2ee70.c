// Function: sub_B2EE70
// Address: 0xb2ee70
//
__int64 __fastcall sub_B2EE70(__int64 a1, __int64 a2, char a3)
{
  __int64 v4; // rax
  __int64 v5; // rbx
  __int64 v6; // r15
  unsigned __int8 v7; // al
  _BYTE *v8; // r14
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdx
  unsigned __int8 v13; // al
  __int64 v14; // r15
  __int64 v15; // rdx
  _QWORD *v16; // rax
  unsigned __int8 v18; // al
  __int64 v19; // r15
  __int64 v20; // rdx
  _QWORD *v21; // rax

  if ( (*(_BYTE *)(a2 + 7) & 0x20) == 0 )
    goto LABEL_19;
  v4 = sub_B91C10(a2, 2);
  v5 = v4;
  if ( !v4 )
    goto LABEL_19;
  v6 = v4 - 16;
  v7 = *(_BYTE *)(v4 - 16);
  if ( (v7 & 2) != 0 )
  {
    v8 = **(_BYTE ***)(v5 - 32);
    if ( !v8 )
      goto LABEL_19;
  }
  else
  {
    v8 = *(_BYTE **)(v6 - 8LL * ((v7 >> 2) & 0xF));
    if ( !v8 )
      goto LABEL_19;
  }
  if ( *v8 )
    goto LABEL_19;
  v9 = sub_B91420(v8, 2);
  if ( v10 == 20
    && !(*(_QWORD *)v9 ^ 0x6E6F6974636E7566LL | *(_QWORD *)(v9 + 8) ^ 0x635F7972746E655FLL)
    && *(_DWORD *)(v9 + 16) == 1953396079 )
  {
    v18 = *(_BYTE *)(v5 - 16);
    if ( (v18 & 2) != 0 )
      v19 = *(_QWORD *)(v5 - 32);
    else
      v19 = v6 - 8LL * ((v18 >> 2) & 0xF);
    v20 = *(_QWORD *)(*(_QWORD *)(v19 + 8) + 136LL);
    v21 = *(_QWORD **)(v20 + 24);
    if ( *(_DWORD *)(v20 + 32) > 0x40u )
      v21 = (_QWORD *)*v21;
    if ( v21 != (_QWORD *)-1LL )
    {
      *(_QWORD *)a1 = v21;
      *(_DWORD *)(a1 + 8) = 0;
      *(_BYTE *)(a1 + 16) = 1;
      return a1;
    }
    goto LABEL_19;
  }
  if ( !a3
    || (v11 = sub_B91420(v8, 2), v12 != 30)
    || *(_QWORD *)v11 ^ 0x69746568746E7973LL | *(_QWORD *)(v11 + 8) ^ 0x6974636E75665F63LL
    || *(_QWORD *)(v11 + 16) != 0x7972746E655F6E6FLL
    || *(_DWORD *)(v11 + 24) != 1970234207
    || *(_WORD *)(v11 + 28) != 29806 )
  {
LABEL_19:
    *(_BYTE *)(a1 + 16) = 0;
    return a1;
  }
  v13 = *(_BYTE *)(v5 - 16);
  if ( (v13 & 2) != 0 )
    v14 = *(_QWORD *)(v5 - 32);
  else
    v14 = v6 - 8LL * ((v13 >> 2) & 0xF);
  v15 = *(_QWORD *)(*(_QWORD *)(v14 + 8) + 136LL);
  v16 = *(_QWORD **)(v15 + 24);
  if ( *(_DWORD *)(v15 + 32) > 0x40u )
    v16 = (_QWORD *)*v16;
  *(_QWORD *)a1 = v16;
  *(_DWORD *)(a1 + 8) = 1;
  *(_BYTE *)(a1 + 16) = 1;
  return a1;
}
