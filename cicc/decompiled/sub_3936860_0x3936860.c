// Function: sub_3936860
// Address: 0x3936860
//
_BYTE *__fastcall sub_3936860(__int64 a1, char *a2, __int64 a3)
{
  char v3; // bl
  __int64 *v4; // rax
  __int64 v5; // rdx
  __int64 *v6; // r13
  _BYTE *v7; // r13
  unsigned __int8 v9; // al
  unsigned __int8 v10; // al
  unsigned __int8 v11; // al
  int v12; // edx

  v3 = (char)a2;
  v4 = sub_1688290(128, (__int64)a2, a3);
  v6 = v4;
  if ( *(_BYTE *)(a1 + 8) )
  {
    sub_16884F0(v4, ".pragma \"");
    if ( (_BYTE)a2 )
      sub_16884F0(v6, "call_");
    v12 = *(_DWORD *)(a1 + 20);
    if ( v12 == -1 )
    {
      sub_1688630(v6, "abi_param_reg all");
      if ( !*(_BYTE *)(a1 + 10) )
        goto LABEL_30;
    }
    else
    {
      sub_1688630(v6, "abi_param_reg %d", v12);
      if ( !*(_BYTE *)(a1 + 10) )
      {
LABEL_30:
        a2 = "\";\n";
        sub_16884F0(v6, "\";\n");
        goto LABEL_2;
      }
    }
    sub_1688630(v6, ", %d", *(_DWORD *)(a1 + 24));
    goto LABEL_30;
  }
LABEL_2:
  if ( *(_BYTE *)(a1 + 12) )
  {
    sub_16884F0(v6, ".pragma \"");
    if ( v3 )
      sub_16884F0(v6, "call_");
    if ( (*(_BYTE *)(a1 + 76) & 1) != 0 )
      sub_16884F0(v6, "retaddr_reg<R:rel32>");
    else
      sub_16884F0(v6, "retaddr_reg<R>");
    sub_1688630(v6, " %d", *(unsigned int *)(a1 + 28));
  }
  else
  {
    if ( !*(_BYTE *)(a1 + 13) )
    {
      if ( !*(_BYTE *)(a1 + 14) )
        goto LABEL_5;
      goto LABEL_12;
    }
    sub_16884F0(v6, ".pragma \"");
    if ( v3 )
      sub_16884F0(v6, "call_");
    sub_1688630(v6, "retaddr_reg<U> %d", *(unsigned int *)(a1 + 32));
  }
  a2 = "\";\n";
  sub_16884F0(v6, "\";\n");
  if ( !*(_BYTE *)(a1 + 14) )
  {
LABEL_5:
    if ( !*(_BYTE *)(a1 + 15) )
      goto LABEL_6;
    goto LABEL_15;
  }
LABEL_12:
  sub_16884F0(v6, ".pragma \"");
  if ( v3 )
    sub_16884F0(v6, "call_");
  sub_16884F0(v6, "scratch_regs<B> ");
  sub_39365E0(v6, *(unsigned int *)(a1 + 36), 0, 0);
  a2 = "\";\n";
  sub_16884F0(v6, "\";\n");
  if ( !*(_BYTE *)(a1 + 15) )
  {
LABEL_6:
    if ( (*(_BYTE *)(a1 + 76) & 2) == 0 )
      goto LABEL_7;
    goto LABEL_18;
  }
LABEL_15:
  sub_16884F0(v6, ".pragma \"");
  if ( v3 )
    sub_16884F0(v6, "call_");
  sub_16884F0(v6, "scratch_regs<R> ");
  v9 = sub_39365E0(v6, *(_QWORD *)(a1 + 64), 0, 0);
  v10 = sub_39365E0(v6, *(_QWORD *)(a1 + 56), 64, v9);
  v11 = sub_39365E0(v6, *(_QWORD *)(a1 + 48), 128, v10);
  sub_39365E0(v6, *(_QWORD *)(a1 + 40), 192, v11);
  a2 = "\";\n";
  sub_16884F0(v6, "\";\n");
  if ( (*(_BYTE *)(a1 + 76) & 2) != 0 )
  {
LABEL_18:
    sub_16884F0(v6, ".pragma \"");
    if ( v3 )
      sub_16884F0(v6, "call_");
    sub_1688630(v6, "allow_conv_alloc");
    a2 = "\";\n";
    sub_16884F0(v6, "\";\n");
  }
LABEL_7:
  v7 = sub_16884C0(v6, (__int64)a2, v5);
  sub_1683B10((__int64)v7, (__int64 *)a1);
  return v7;
}
