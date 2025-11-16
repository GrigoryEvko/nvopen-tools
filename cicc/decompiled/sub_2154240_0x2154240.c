// Function: sub_2154240
// Address: 0x2154240
//
void __fastcall sub_2154240(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int8 v6; // al
  const char *v7; // rsi
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax
  int v11; // ecx

  v6 = *(_BYTE *)(a2 + 16);
  if ( v6 == 13 )
  {
    sub_16A95F0(a2 + 24, a3, 1);
    return;
  }
  if ( v6 == 14 )
  {
    sub_21523A0(a1, (__int64 *)a2, a3);
    return;
  }
  v7 = "0";
  if ( v6 == 15 )
    goto LABEL_11;
  if ( v6 > 3u )
  {
    v10 = sub_2153350(a1, a2, 0, a4);
    sub_21537F0(a1, v10, a3, v11);
  }
  else
  {
    if ( !(*(_DWORD *)(*(_QWORD *)a2 + 8LL) >> 8) && *(_BYTE *)(a1 + 896) && v6 )
    {
      sub_1263B40(a3, "generic(");
      v8 = sub_396EAF0(a1, a2);
      sub_38E2490(v8, a3, *(_QWORD *)(a1 + 240));
      v7 = ")";
LABEL_11:
      sub_1263B40(a3, v7);
      return;
    }
    v9 = sub_396EAF0(a1, a2);
    sub_38E2490(v9, a3, *(_QWORD *)(a1 + 240));
  }
}
