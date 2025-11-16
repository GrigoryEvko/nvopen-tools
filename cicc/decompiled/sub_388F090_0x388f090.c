// Function: sub_388F090
// Address: 0x388f090
//
__int64 __fastcall sub_388F090(__int64 a1, _DWORD *a2)
{
  unsigned __int64 v2; // rsi
  unsigned int v4; // r12d
  bool v5; // al
  const char *v6; // [rsp+10h] [rbp-30h] BYREF
  char v7; // [rsp+20h] [rbp-20h]
  char v8; // [rsp+21h] [rbp-1Fh]

  if ( *(_DWORD *)(a1 + 64) == 390 && *(_BYTE *)(a1 + 164) )
  {
    v4 = *(_DWORD *)(a1 + 160);
    if ( v4 <= 0x40 )
      v5 = *(_QWORD *)(a1 + 152) == 0;
    else
      v5 = v4 == (unsigned int)sub_16A57B0(a1 + 152);
    *a2 = !v5;
    *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
    return 0;
  }
  else
  {
    v2 = *(_QWORD *)(a1 + 56);
    v8 = 1;
    v7 = 3;
    v6 = "expected integer";
    return sub_38814C0(a1 + 8, v2, (__int64)&v6);
  }
}
