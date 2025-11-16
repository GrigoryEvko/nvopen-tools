// Function: sub_990670
// Address: 0x990670
//
bool __fastcall sub_990670(__int64 a1, _QWORD *a2, _QWORD *a3)
{
  __int64 v3; // rax
  int v7; // eax
  __int64 v8; // r14
  __int64 v10; // rdx
  __int64 v11; // rdi
  __int64 v12; // rdi
  __int64 v13; // rax
  __int64 v14; // rsi
  _BYTE *v15; // rax
  _BYTE *v16; // rax

  v3 = *(_QWORD *)(a1 - 32);
  if ( !v3 || *(_BYTE *)v3 || *(_QWORD *)(v3 + 24) != *(_QWORD *)(a1 + 80) )
    BUG();
  v7 = sub_9905C0(*(_DWORD *)(v3 + 36));
  v8 = *(_QWORD *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
  if ( *(_BYTE *)v8 != 85 )
    return 0;
  v10 = *(_QWORD *)(v8 - 32);
  if ( !v10
    || *(_BYTE *)v10
    || *(_QWORD *)(v10 + 24) != *(_QWORD *)(v8 + 80)
    || (*(_BYTE *)(v10 + 33) & 0x20) == 0
    || v7 != *(_DWORD *)(v10 + 36) )
  {
    return 0;
  }
  v11 = *(_QWORD *)(a1 + 32 * (1LL - (*(_DWORD *)(a1 + 4) & 0x7FFFFFF)));
  if ( *(_BYTE *)v11 == 17 )
  {
    *a2 = v11 + 24;
  }
  else
  {
    if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v11 + 8) + 8LL) - 17 > 1 )
      return 0;
    if ( *(_BYTE *)v11 > 0x15u )
      return 0;
    v15 = (_BYTE *)sub_AD7630(v11, 0);
    if ( !v15 || *v15 != 17 )
      return 0;
    *a2 = v15 + 24;
  }
  v12 = *(_QWORD *)(v8 + 32 * (1LL - (*(_DWORD *)(v8 + 4) & 0x7FFFFFF)));
  if ( *(_BYTE *)v12 == 17 )
  {
    *a3 = v12 + 24;
    goto LABEL_15;
  }
  if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v12 + 8) + 8LL) - 17 > 1 )
    return 0;
  if ( *(_BYTE *)v12 > 0x15u )
    return 0;
  v16 = (_BYTE *)sub_AD7630(v12, 0);
  if ( !v16 || *v16 != 17 )
    return 0;
  *a3 = v16 + 24;
LABEL_15:
  v13 = *(_QWORD *)(a1 - 32);
  if ( !v13 || *(_BYTE *)v13 || *(_QWORD *)(v13 + 24) != *(_QWORD *)(a1 + 80) )
    BUG();
  v14 = *a2;
  if ( *(_DWORD *)(v13 + 36) == 330 )
  {
    *a2 = *a3;
    *a3 = v14;
  }
  else
  {
    v14 = *a3;
  }
  return (int)sub_C4C880(*a2, v14) <= 0;
}
