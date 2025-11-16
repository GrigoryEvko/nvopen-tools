// Function: sub_E521E0
// Address: 0xe521e0
//
_BYTE *__fastcall sub_E521E0(__int64 a1, unsigned __int64 a2, __int64 a3, char a4, int a5, unsigned int a6)
{
  unsigned __int64 v7; // r12
  __int64 v8; // rdi
  unsigned __int64 v9; // rsi
  char v10; // bl
  __int64 v12; // rdi
  _BYTE *v13; // rax
  unsigned __int64 v15; // r12
  __int64 v16; // rax

  if ( *(_BYTE *)(*(_QWORD *)(a1 + 312) + 21LL) )
  {
    if ( !a2 || (a2 & (a2 - 1)) != 0 )
      sub_C64ED0("Only power-of-two alignments are supported with .align.", 1u);
    _BitScanReverse64(&v7, a2);
    sub_904010(*(_QWORD *)(a1 + 304), "\t.align\t");
    v8 = *(_QWORD *)(a1 + 304);
    v9 = (int)(63 - (v7 ^ 0x3F));
    goto LABEL_5;
  }
  v10 = a5;
  if ( a2 && (a2 & (a2 - 1)) == 0 )
  {
    switch ( a5 )
    {
      case 2:
        sub_904010(*(_QWORD *)(a1 + 304), ".p2alignw ");
LABEL_20:
        _BitScanReverse64(&v15, a2);
        sub_CB59D0(*(_QWORD *)(a1 + 304), (int)(63 - (v15 ^ 0x3F)));
        if ( a4 )
        {
          sub_904010(*(_QWORD *)(a1 + 304), ", 0x");
          sub_CB5A50(*(_QWORD *)(a1 + 304), a3 & (0xFFFFFFFFFFFFFFFFLL >> (8 * (8 - v10))));
          if ( !a6 )
            return sub_E4D880(a1);
          goto LABEL_22;
        }
LABEL_13:
        if ( !a6 )
          return sub_E4D880(a1);
        sub_904010(*(_QWORD *)(a1 + 304), ", ");
        goto LABEL_22;
      case 4:
        sub_904010(*(_QWORD *)(a1 + 304), ".p2alignl ");
        goto LABEL_20;
      case 1:
        sub_904010(*(_QWORD *)(a1 + 304), "\t.p2align\t");
        goto LABEL_20;
    }
LABEL_34:
    BUG();
  }
  switch ( a5 )
  {
    case 2:
      sub_904010(*(_QWORD *)(a1 + 304), ".balignw");
      break;
    case 4:
      sub_904010(*(_QWORD *)(a1 + 304), ".balignl");
      break;
    case 1:
      sub_904010(*(_QWORD *)(a1 + 304), ".balign");
      break;
    default:
      goto LABEL_34;
  }
  v12 = *(_QWORD *)(a1 + 304);
  v13 = *(_BYTE **)(v12 + 32);
  if ( (unsigned __int64)v13 >= *(_QWORD *)(v12 + 24) )
  {
    v12 = sub_CB5D20(v12, 32);
  }
  else
  {
    *(_QWORD *)(v12 + 32) = v13 + 1;
    *v13 = 32;
  }
  sub_CB59D0(v12, a2);
  if ( !a4 )
    goto LABEL_13;
  v16 = sub_904010(*(_QWORD *)(a1 + 304), ", ");
  sub_CB59F0(v16, a3 & (0xFFFFFFFFFFFFFFFFLL >> (8 * (8 - v10))));
  if ( !a6 )
    return sub_E4D880(a1);
LABEL_22:
  v9 = a6;
  v8 = sub_904010(*(_QWORD *)(a1 + 304), ", ");
LABEL_5:
  sub_CB59D0(v8, v9);
  return sub_E4D880(a1);
}
