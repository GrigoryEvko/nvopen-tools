// Function: sub_388BA90
// Address: 0x388ba90
//
__int64 __fastcall sub_388BA90(__int64 a1, _DWORD *a2)
{
  unsigned __int64 v3; // rsi
  unsigned int v5; // r12d
  __int64 v6; // rdi
  unsigned __int64 v7; // rsi
  unsigned __int64 v8; // rax
  _QWORD v9[2]; // [rsp+10h] [rbp-30h] BYREF
  char v10; // [rsp+20h] [rbp-20h]
  char v11; // [rsp+21h] [rbp-1Fh]

  if ( *(_DWORD *)(a1 + 64) != 390 || !*(_BYTE *)(a1 + 164) )
  {
    v3 = *(_QWORD *)(a1 + 56);
    v11 = 1;
    v10 = 3;
    v9[0] = "expected integer";
    return sub_38814C0(a1 + 8, v3, (__int64)v9);
  }
  v5 = *(_DWORD *)(a1 + 160);
  if ( v5 <= 0x40 )
  {
    v8 = *(_QWORD *)(a1 + 152);
    v6 = a1 + 8;
    if ( v8 > 0x100000000LL )
      goto LABEL_7;
  }
  else if ( v5 - (unsigned int)sub_16A57B0(a1 + 152) > 0x40 || (v8 = **(_QWORD **)(a1 + 152), v8 > 0x100000000LL) )
  {
    v6 = a1 + 8;
LABEL_7:
    v7 = *(_QWORD *)(a1 + 56);
    v11 = 1;
    v9[0] = "expected 32-bit integer (too large)";
    v10 = 3;
    return sub_38814C0(v6, v7, (__int64)v9);
  }
  v6 = a1 + 8;
  if ( (unsigned int)v8 != v8 )
    goto LABEL_7;
  *a2 = v8;
  *(_DWORD *)(a1 + 64) = sub_3887100(v6);
  return 0;
}
