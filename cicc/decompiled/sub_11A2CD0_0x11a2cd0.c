// Function: sub_11A2CD0
// Address: 0x11a2cd0
//
__int64 __fastcall sub_11A2CD0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  unsigned int v3; // r12d
  __int64 v5; // rdx
  char *v6; // r14
  char v7; // al
  __int64 v8; // rax
  unsigned int v9; // eax
  __int64 v10; // rax
  __int64 v11; // r14
  unsigned int v12; // r15d
  _QWORD *v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rbx
  unsigned int v16; // r14d
  _QWORD *v17; // rax
  __int64 v18; // rdx

  if ( *(_BYTE *)a2 != 85 )
    return 0;
  v2 = *(_QWORD *)(a2 - 32);
  if ( !v2 || *(_BYTE *)v2 || *(_QWORD *)(v2 + 24) != *(_QWORD *)(a2 + 80) || *(_DWORD *)(v2 + 36) != *(_DWORD *)a1 )
    return 0;
  v5 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  v6 = *(char **)(a2 + 32 * (*(unsigned int *)(a1 + 8) - v5));
  v7 = *v6;
  if ( (unsigned __int8)*v6 <= 0x1Cu )
  {
    if ( v7 != 5 || *((_WORD *)v6 + 1) != 34 )
      return 0;
  }
  else if ( v7 != 63 )
  {
    return 0;
  }
  v8 = sub_BB5290(*(_QWORD *)(a2 + 32 * (*(unsigned int *)(a1 + 8) - v5)));
  LOBYTE(v9) = sub_BCAC40(v8, 8);
  v3 = v9;
  if ( !(_BYTE)v9 )
    return 0;
  v10 = *(_QWORD *)&v6[-32 * (*((_DWORD *)v6 + 1) & 0x7FFFFFF)];
  if ( !v10 )
    return 0;
  **(_QWORD **)(a1 + 16) = v10;
  v11 = *(_QWORD *)&v6[32 * (1LL - (*((_DWORD *)v6 + 1) & 0x7FFFFFF))];
  if ( *(_BYTE *)v11 != 17 )
    return 0;
  v12 = *(_DWORD *)(v11 + 32);
  if ( v12 > 0x40 )
  {
    if ( v12 - (unsigned int)sub_C444A0(v11 + 24) > 0x40 )
      return 0;
    v13 = *(_QWORD **)(a1 + 24);
    v14 = **(_QWORD **)(v11 + 24);
  }
  else
  {
    v13 = *(_QWORD **)(a1 + 24);
    v14 = *(_QWORD *)(v11 + 24);
  }
  *v13 = v14;
  if ( *(_BYTE *)a2 == 85 )
  {
    v15 = *(_QWORD *)(a2 + 32 * (*(unsigned int *)(a1 + 32) - (unsigned __int64)(*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
    if ( *(_BYTE *)v15 == 17 )
    {
      v16 = *(_DWORD *)(v15 + 32);
      if ( v16 <= 0x40 )
      {
        v17 = *(_QWORD **)(a1 + 40);
        v18 = *(_QWORD *)(v15 + 24);
LABEL_19:
        *v17 = v18;
        return v3;
      }
      if ( v16 - (unsigned int)sub_C444A0(v15 + 24) <= 0x40 )
      {
        v17 = *(_QWORD **)(a1 + 40);
        v18 = **(_QWORD **)(v15 + 24);
        goto LABEL_19;
      }
    }
  }
  return 0;
}
