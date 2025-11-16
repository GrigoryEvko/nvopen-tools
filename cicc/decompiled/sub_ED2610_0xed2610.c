// Function: sub_ED2610
// Address: 0xed2610
//
__int64 __fastcall sub_ED2610(__int64 a1, int a2)
{
  __int64 v2; // rax
  __int64 v3; // r12
  unsigned __int8 v4; // al
  __int64 *v5; // rdx
  __int64 v6; // r13
  _WORD *v7; // rax
  __int64 v8; // rdx
  unsigned __int8 v9; // al
  __int64 v10; // r13
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rax

  if ( (*(_BYTE *)(a1 + 7) & 0x20) == 0 )
    return 0;
  v2 = sub_B91C10(a1, 2);
  v3 = v2;
  if ( !v2 )
    return 0;
  v4 = *(_BYTE *)(v2 - 16);
  if ( (v4 & 2) != 0 )
  {
    if ( *(_DWORD *)(v3 - 24) > 4u )
    {
      v5 = *(__int64 **)(v3 - 32);
      v6 = v3 - 16;
      goto LABEL_6;
    }
    return 0;
  }
  if ( ((*(_WORD *)(v3 - 16) >> 6) & 0xFu) <= 4 )
    return 0;
  v6 = v3 - 16;
  v5 = (__int64 *)(v3 - 16 - 8LL * ((v4 >> 2) & 0xF));
LABEL_6:
  if ( !*v5 )
    return 0;
  v7 = (_WORD *)sub_B91420(*v5);
  if ( v8 != 2 || *v7 != 20566 )
    return 0;
  v9 = *(_BYTE *)(v3 - 16);
  v10 = (v9 & 2) != 0 ? *(_QWORD *)(v3 - 32) : v6 - 8LL * ((v9 >> 2) & 0xF);
  v11 = *(_QWORD *)(v10 + 8);
  if ( *(_BYTE *)v11 != 1 )
    return 0;
  v12 = *(_QWORD *)(v11 + 136);
  if ( *(_BYTE *)v12 != 17 )
    return 0;
  v13 = *(_DWORD *)(v12 + 32) <= 0x40u ? *(_QWORD *)(v12 + 24) : **(_QWORD **)(v12 + 24);
  if ( a2 != v13 )
    return 0;
  return v3;
}
