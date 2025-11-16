// Function: sub_B2CA40
// Address: 0xb2ca40
//
__int64 __fastcall sub_B2CA40(__int64 a1, char a2)
{
  __int64 v3; // r12
  __int64 v4; // rbx
  __int64 v5; // rdi
  _QWORD *v6; // rdi
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v12; // rax
  __int64 **v13; // rax
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // rsi
  __int64 v18; // rcx
  __int64 v19; // rdi
  _QWORD *v20; // rdx
  __int64 v21; // rsi
  __int64 v22; // rsi
  __int64 v23; // rdx
  __int64 v24; // rsi
  __int64 v25; // rcx
  __int64 v26; // rcx
  __int64 v27; // rcx

  v3 = a1 + 72;
  v4 = *(_QWORD *)(a1 + 80);
  *(_WORD *)(a1 + 34) &= ~0x800u;
  if ( v4 == a1 + 72 )
    goto LABEL_9;
  do
  {
    v5 = v4 - 24;
    if ( !v4 )
      v5 = 0;
    sub_AA5200(v5);
    v4 = *(_QWORD *)(v4 + 8);
  }
  while ( v3 != v4 );
  while ( v3 != (*(_QWORD *)(a1 + 72) & 0xFFFFFFFFFFFFFFF8LL) )
  {
    v6 = *(_QWORD **)(a1 + 80);
    if ( v6 )
      v6 -= 3;
    sub_AA5450(v6);
LABEL_9:
    ;
  }
  if ( (*(_DWORD *)(a1 + 4) & 0x7FFFFFF) != 0 )
  {
    if ( a2 )
    {
      v7 = 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
      if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
      {
        v8 = *(_QWORD *)(a1 - 8);
        v9 = v8 + v7;
      }
      else
      {
        v8 = a1 - v7;
        v9 = a1;
      }
      do
      {
        if ( *(_QWORD *)v8 )
        {
          v10 = *(_QWORD *)(v8 + 8);
          **(_QWORD **)(v8 + 16) = v10;
          if ( v10 )
            *(_QWORD *)(v10 + 16) = *(_QWORD *)(v8 + 16);
        }
        *(_QWORD *)v8 = 0;
        v8 += 32;
      }
      while ( v9 != v8 );
      *(_DWORD *)(a1 + 4) &= 0xF8000000;
      *(_WORD *)(a1 + 2) &= 0xFFF1u;
      return sub_B91E30(a1);
    }
    v12 = sub_B2BE50(a1);
    v13 = (__int64 **)sub_BCE3C0(v12, 0);
    v14 = sub_AC9EC0(v13);
    v15 = *(_QWORD *)(a1 - 8);
    if ( *(_QWORD *)v15 )
    {
      v16 = *(_QWORD *)(v15 + 8);
      **(_QWORD **)(v15 + 16) = v16;
      if ( v16 )
        *(_QWORD *)(v16 + 16) = *(_QWORD *)(v15 + 16);
    }
    *(_QWORD *)v15 = v14;
    if ( v14 )
    {
      v17 = *(_QWORD *)(v14 + 16);
      v18 = v14 + 16;
      v19 = v14 + 16;
      *(_QWORD *)(v15 + 8) = v17;
      if ( v17 )
        *(_QWORD *)(v17 + 16) = v15 + 8;
      *(_QWORD *)(v15 + 16) = v18;
      *(_QWORD *)(v14 + 16) = v15;
      v20 = *(_QWORD **)(a1 - 8);
      if ( v20[4] && (v21 = v20[5], (*(_QWORD *)v20[6] = v21) != 0) )
      {
        *(_QWORD *)(v21 + 16) = v20[6];
        v19 = v14 + 16;
        v20[4] = v14;
      }
      else
      {
        v20[4] = v14;
      }
      v22 = *(_QWORD *)(v14 + 16);
      v20[5] = v22;
      if ( v22 )
        *(_QWORD *)(v22 + 16) = v20 + 5;
      v20[6] = v18;
      *(_QWORD *)(v14 + 16) = v20 + 4;
      v23 = *(_QWORD *)(a1 - 8);
      v24 = v23 + 64;
      if ( !*(_QWORD *)(v23 + 64) || (v25 = *(_QWORD *)(v23 + 72), (**(_QWORD **)(v23 + 80) = v25) == 0) )
      {
        *(_QWORD *)(v23 + 64) = v14;
        goto LABEL_35;
      }
    }
    else
    {
      v23 = *(_QWORD *)(a1 - 8);
      if ( *(_QWORD *)(v23 + 32) )
      {
        v27 = *(_QWORD *)(v23 + 40);
        **(_QWORD **)(v23 + 48) = v27;
        if ( v27 )
          *(_QWORD *)(v27 + 16) = *(_QWORD *)(v23 + 48);
        *(_QWORD *)(v23 + 32) = 0;
        v23 = *(_QWORD *)(a1 - 8);
      }
      if ( !*(_QWORD *)(v23 + 64) )
        goto LABEL_38;
      v25 = *(_QWORD *)(v23 + 72);
      v24 = v23 + 64;
      **(_QWORD **)(v23 + 80) = v25;
      if ( !v25 )
      {
        *(_QWORD *)(v23 + 64) = 0;
        goto LABEL_38;
      }
    }
    *(_QWORD *)(v25 + 16) = *(_QWORD *)(v23 + 80);
    *(_QWORD *)(v23 + 64) = v14;
    if ( v14 )
    {
      v19 = v14 + 16;
LABEL_35:
      v26 = *(_QWORD *)(v14 + 16);
      *(_QWORD *)(v23 + 72) = v26;
      if ( v26 )
        *(_QWORD *)(v26 + 16) = v23 + 72;
      *(_QWORD *)(v23 + 80) = v19;
      *(_QWORD *)(v14 + 16) = v24;
    }
LABEL_38:
    *(_WORD *)(a1 + 2) &= 0xFFF1u;
  }
  return sub_B91E30(a1);
}
