// Function: sub_B2E530
// Address: 0xb2e530
//
void __fastcall sub_B2E530(__int64 a1)
{
  __int64 v2; // rax
  __int64 **v3; // rax
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // rsi
  __int64 v8; // rcx
  __int64 v9; // rdi
  _QWORD *v10; // rdx
  __int64 v11; // rsi
  __int64 v12; // rsi
  __int64 v13; // rdx
  __int64 v14; // rsi
  __int64 v15; // rcx
  __int64 v16; // rcx
  __int64 v17; // rcx

  if ( (*(_DWORD *)(a1 + 4) & 0x7FFFFFF) != 0 )
    return;
  sub_BD2A10(a1, 3, 0);
  *(_DWORD *)(a1 + 4) = *(_DWORD *)(a1 + 4) & 0xF8000000 | 3;
  v2 = sub_B2BE50(a1);
  v3 = (__int64 **)sub_BCE3C0(v2, 0);
  v4 = sub_AC9EC0(v3);
  v5 = *(_QWORD *)(a1 - 8);
  if ( *(_QWORD *)v5 )
  {
    v6 = *(_QWORD *)(v5 + 8);
    **(_QWORD **)(v5 + 16) = v6;
    if ( v6 )
      *(_QWORD *)(v6 + 16) = *(_QWORD *)(v5 + 16);
  }
  *(_QWORD *)v5 = v4;
  if ( v4 )
  {
    v7 = *(_QWORD *)(v4 + 16);
    v8 = v4 + 16;
    v9 = v4 + 16;
    *(_QWORD *)(v5 + 8) = v7;
    if ( v7 )
      *(_QWORD *)(v7 + 16) = v5 + 8;
    *(_QWORD *)(v5 + 16) = v8;
    *(_QWORD *)(v4 + 16) = v5;
    v10 = *(_QWORD **)(a1 - 8);
    if ( v10[4] && (v11 = v10[5], (*(_QWORD *)v10[6] = v11) != 0) )
    {
      *(_QWORD *)(v11 + 16) = v10[6];
      v9 = v4 + 16;
      v10[4] = v4;
    }
    else
    {
      v10[4] = v4;
    }
    v12 = *(_QWORD *)(v4 + 16);
    v10[5] = v12;
    if ( v12 )
      *(_QWORD *)(v12 + 16) = v10 + 5;
    v10[6] = v8;
    *(_QWORD *)(v4 + 16) = v10 + 4;
    v13 = *(_QWORD *)(a1 - 8);
    v14 = v13 + 64;
    if ( !*(_QWORD *)(v13 + 64) || (v15 = *(_QWORD *)(v13 + 72), (**(_QWORD **)(v13 + 80) = v15) == 0) )
    {
      *(_QWORD *)(v13 + 64) = v4;
LABEL_17:
      v16 = *(_QWORD *)(v4 + 16);
      *(_QWORD *)(v13 + 72) = v16;
      if ( v16 )
        *(_QWORD *)(v16 + 16) = v13 + 72;
      *(_QWORD *)(v13 + 80) = v9;
      *(_QWORD *)(v4 + 16) = v14;
      return;
    }
LABEL_15:
    *(_QWORD *)(v15 + 16) = *(_QWORD *)(v13 + 80);
    *(_QWORD *)(v13 + 64) = v4;
    if ( !v4 )
      return;
    v9 = v4 + 16;
    goto LABEL_17;
  }
  v13 = *(_QWORD *)(a1 - 8);
  if ( *(_QWORD *)(v13 + 32) )
  {
    v17 = *(_QWORD *)(v13 + 40);
    **(_QWORD **)(v13 + 48) = v17;
    if ( v17 )
      *(_QWORD *)(v17 + 16) = *(_QWORD *)(v13 + 48);
    *(_QWORD *)(v13 + 32) = 0;
    v13 = *(_QWORD *)(a1 - 8);
  }
  if ( *(_QWORD *)(v13 + 64) )
  {
    v15 = *(_QWORD *)(v13 + 72);
    v14 = v13 + 64;
    **(_QWORD **)(v13 + 80) = v15;
    if ( !v15 )
    {
      *(_QWORD *)(v13 + 64) = 0;
      return;
    }
    goto LABEL_15;
  }
}
