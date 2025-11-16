// Function: sub_388BD80
// Address: 0x388bd80
//
__int64 __fastcall sub_388BD80(__int64 a1, __int64 *a2)
{
  unsigned __int64 v2; // rsi
  unsigned int v4; // r12d
  __int64 v5; // rax
  unsigned int v6; // r12d
  const char *v7; // [rsp+10h] [rbp-30h] BYREF
  char v8; // [rsp+20h] [rbp-20h]
  char v9; // [rsp+21h] [rbp-1Fh]

  if ( *(_DWORD *)(a1 + 64) == 390 && *(_BYTE *)(a1 + 164) )
  {
    v4 = *(_DWORD *)(a1 + 160);
    if ( v4 > 0x40 )
    {
      v6 = v4 - sub_16A57B0(a1 + 152);
      v5 = -1;
      if ( v6 <= 0x40 )
        v5 = **(_QWORD **)(a1 + 152);
    }
    else
    {
      v5 = *(_QWORD *)(a1 + 152);
    }
    *a2 = v5;
    *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
    return 0;
  }
  else
  {
    v2 = *(_QWORD *)(a1 + 56);
    v9 = 1;
    v8 = 3;
    v7 = "expected integer";
    return sub_38814C0(a1 + 8, v2, (__int64)&v7);
  }
}
