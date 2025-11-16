// Function: sub_1E74580
// Address: 0x1e74580
//
void __fastcall sub_1E74580(__int64 a1, _QWORD *a2, char a3)
{
  __int64 *v3; // rcx
  __int64 v5; // rax
  __int64 *v6; // r15
  __int64 v7; // rbx
  __int64 v8; // r12
  __int64 v9; // rsi
  unsigned __int64 v10; // rax

  v3 = a2 + 4;
  v5 = a2[1];
  v6 = (__int64 *)v5;
  if ( !a3 )
  {
    if ( !v5 )
      BUG();
    if ( (*(_BYTE *)v5 & 4) != 0 )
    {
      v6 = *(__int64 **)(v5 + 8);
      v3 = a2 + 14;
    }
    else
    {
      while ( (*(_BYTE *)(v5 + 46) & 8) != 0 )
        v5 = *(_QWORD *)(v5 + 8);
      v6 = *(__int64 **)(v5 + 8);
      v3 = a2 + 14;
    }
  }
  v7 = *v3;
  v8 = *v3 + 16LL * *((unsigned int *)v3 + 2);
  if ( *v3 != v8 )
  {
    do
    {
      while ( 1 )
      {
        if ( (*(_QWORD *)v7 & 6) != 0 || *(int *)(v7 + 8) <= 0 )
          goto LABEL_7;
        v10 = *(_QWORD *)v7 & 0xFFFFFFFFFFFFFFF8LL;
        if ( !a3 )
          break;
        if ( *(_DWORD *)(v10 + 120) <= 1u )
          goto LABEL_5;
LABEL_7:
        v7 += 16;
        if ( v8 == v7 )
          return;
      }
      if ( *(_DWORD *)(v10 + 40) <= 1u )
      {
LABEL_5:
        v9 = *(_QWORD *)(v10 + 8);
        if ( **(_WORD **)(v9 + 16) == 15 )
          sub_1E705C0(*(_QWORD *)(a1 + 128), (unsigned __int64 *)v9, v6);
        goto LABEL_7;
      }
      v7 += 16;
    }
    while ( v8 != v7 );
  }
}
