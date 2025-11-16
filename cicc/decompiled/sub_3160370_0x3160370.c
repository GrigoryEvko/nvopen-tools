// Function: sub_3160370
// Address: 0x3160370
//
void __fastcall sub_3160370(__int64 a1)
{
  _QWORD *v1; // r12
  __int64 *v2; // rax
  __int64 **v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rcx
  __int64 v7; // rcx
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rcx

  v1 = *(_QWORD **)(a1 + 32 * (1LL - (*(_DWORD *)(a1 + 4) & 0x7FFFFFF)));
  v2 = (__int64 *)sub_BD5C60(a1);
  v3 = (__int64 **)sub_BCE3C0(v2, 0);
  v4 = sub_AC9EC0(v3);
  v5 = a1 + 32 * (1LL - (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
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
    *(_QWORD *)(v5 + 8) = v7;
    if ( v7 )
      *(_QWORD *)(v7 + 16) = v5 + 8;
    *(_QWORD *)(v5 + 16) = v4 + 16;
    *(_QWORD *)(v4 + 16) = v5;
  }
  if ( *(_BYTE *)v1 != 60 )
  {
    if ( v1[2] )
    {
      v8 = *(_QWORD *)(a1 + 16);
      if ( !v8 )
LABEL_22:
        BUG();
      while ( 1 )
      {
        v9 = *(_QWORD *)(v8 + 24);
        if ( *(_BYTE *)v9 == 85 )
        {
          v10 = *(_QWORD *)(v9 - 32);
          if ( v10 )
          {
            if ( !*(_BYTE *)v10
              && *(_QWORD *)(v10 + 24) == *(_QWORD *)(v9 + 80)
              && (*(_BYTE *)(v10 + 33) & 0x20) != 0
              && (unsigned int)(*(_DWORD *)(v10 + 36) - 39) <= 1 )
            {
              break;
            }
          }
        }
        v8 = *(_QWORD *)(v8 + 8);
        if ( !v8 )
          goto LABEL_22;
      }
      sub_B444E0(v1, *(_QWORD *)(v9 + 32), 0);
    }
    else
    {
      sub_B43D60(v1);
    }
  }
}
