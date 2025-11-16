// Function: sub_3027EB0
// Address: 0x3027eb0
//
void __fastcall sub_3027EB0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int8 v5; // al
  const char *v6; // rsi
  __int64 v7; // rax
  _BYTE *v8; // rax
  __int64 v9; // rax

  v5 = *(_BYTE *)a2;
  if ( *(_BYTE *)a2 == 17 )
  {
    sub_C49420(a2 + 24, a3, 1);
    return;
  }
  if ( v5 == 18 )
  {
    sub_3026430(a1, a2, a3);
    return;
  }
  v6 = "0";
  if ( v5 == 20 )
  {
LABEL_13:
    sub_904010(a3, v6);
    return;
  }
  if ( v5 <= 3u )
  {
    if ( (*(_BYTE *)(a1 + 1192) & (*(_DWORD *)(*(_QWORD *)(a2 + 8) + 8LL) >> 8 == 0)) == 0 || !v5 )
    {
      v7 = sub_31DB510(a1, a2);
      sub_EA12C0(v7, a3, *(_BYTE **)(a1 + 208));
      return;
    }
    sub_904010(a3, "generic(");
    v9 = sub_31DB510(a1, a2);
    sub_EA12C0(v9, a3, *(_BYTE **)(a1 + 208));
    v6 = ")";
    goto LABEL_13;
  }
  if ( v5 != 5 )
    BUG();
  v8 = (_BYTE *)sub_30270A0(a1, (unsigned __int8 *)a2, 0);
  sub_30275E0(a1, v8, a3);
}
