// Function: sub_2E891D0
// Address: 0x2e891d0
//
__int64 __fastcall sub_2E891D0(__int64 a1, __int64 a2)
{
  __int64 v5; // rbx
  __int16 v6; // ax
  int v7; // ecx
  __int16 v8; // dx
  int v9; // esi
  unsigned int i; // r14d
  __int64 v11; // rdi
  __int64 v12; // rax
  __int64 v13; // rsi
  __int64 v14; // rdi
  char v15; // r14
  char v16; // r15
  __int64 v17; // r13
  __int64 v18; // rax
  _BYTE *v19; // rax
  _BYTE *v20; // rax

  if ( (unsigned __int16)(*(_WORD *)(a1 + 68) - 14) > 2u )
    return 0;
  if ( (unsigned __int16)(*(_WORD *)(a2 + 68) - 14) > 2u )
    return 0;
  if ( *(_QWORD *)(a2 + 56) != *(_QWORD *)(a1 + 56) )
    return 0;
  v5 = sub_2E89170(a1);
  if ( v5 != sub_2E89170(a2) )
    return 0;
  v6 = *(_WORD *)(a1 + 68);
  v7 = 1;
  if ( v6 != 14 )
    v7 = -858993459 * ((40LL * (*(_DWORD *)(a1 + 40) & 0xFFFFFF) - 80) >> 3);
  v8 = *(_WORD *)(a2 + 68);
  v9 = 1;
  if ( v8 != 14 )
    v9 = -858993459 * ((40LL * (*(_DWORD *)(a2 + 40) & 0xFFFFFF) - 80) >> 3);
  if ( v9 != v7 )
    return 0;
  for ( i = 0; ; ++i )
  {
    v11 = *(_QWORD *)(a1 + 32);
    if ( v6 == 14 )
      break;
    v11 += 80;
    if ( i >= -858993459 * (unsigned int)((40LL * (*(_DWORD *)(a1 + 40) & 0xFFFFFF) - 80) >> 3) )
      goto LABEL_20;
LABEL_14:
    v12 = 40LL * i;
    v13 = *(_QWORD *)(a2 + 32);
    v14 = v12 + v11;
    if ( v8 != 14 )
      v13 += 80;
    if ( !(unsigned __int8)sub_2EAB6C0(v14, v12 + v13) )
      return 0;
    v6 = *(_WORD *)(a1 + 68);
    v8 = *(_WORD *)(a2 + 68);
  }
  if ( !i )
    goto LABEL_14;
LABEL_20:
  v15 = 0;
  if ( v8 == 14 )
  {
    v19 = *(_BYTE **)(a2 + 32);
    if ( v19[40] == 1 )
      v15 = *v19 == 0;
  }
  v16 = 0;
  v17 = sub_2E891C0(a2);
  if ( *(_WORD *)(a1 + 68) == 14 )
  {
    v20 = *(_BYTE **)(a1 + 32);
    if ( v20[40] == 1 )
      v16 = *v20 == 0;
  }
  v18 = sub_2E891C0(a1);
  return sub_AF65F0(v18, v16, v17, v15);
}
