// Function: sub_F34590
// Address: 0xf34590
//
__int64 __fastcall sub_F34590(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v4; // r12
  __int64 v5; // rsi
  __int64 v7; // rax

  v2 = *(_QWORD *)(a1 + 56);
  if ( !v2 )
    BUG();
  if ( *(_BYTE *)(v2 - 24) != 84 )
    return 0;
  while ( *(_BYTE *)(v2 - 24) == 84 )
  {
    v4 = v2 - 24;
    v5 = **(_QWORD **)(v2 - 32);
    if ( v5 && v4 == v5 )
    {
      v7 = sub_ACADE0(*(__int64 ***)(v4 + 8));
      sub_BD84D0(v4, v7);
      if ( !a2 )
        goto LABEL_8;
LABEL_7:
      sub_1031600(a2, v4);
      goto LABEL_8;
    }
    sub_BD84D0(v4, v5);
    if ( a2 )
      goto LABEL_7;
LABEL_8:
    sub_B43D60((_QWORD *)v4);
    v2 = *(_QWORD *)(a1 + 56);
    if ( !v2 )
      BUG();
  }
  return 1;
}
