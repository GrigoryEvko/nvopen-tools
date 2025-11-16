// Function: sub_14A5330
// Address: 0x14a5330
//
__int64 __fastcall sub_14A5330(__int64 **a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 *v5; // rdi
  __int64 (__fastcall *v6)(__int64, __int64, __int64, __int64); // rax
  unsigned __int8 v7; // al
  __int64 *v8; // r12
  __int64 v10; // r13
  __int64 v11; // rbx
  __int64 **v12; // rax
  __int64 *v13; // r14
  __int64 v14; // rax
  __int64 v15; // [rsp+0h] [rbp-30h]

  v5 = *a1;
  v6 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*v5 + 104);
  if ( v6 != sub_14A5230 )
    return ((__int64 (__fastcall *)(__int64 *, __int64, __int64))v6)(v5, a2, a3);
  v7 = *(_BYTE *)(a2 + 16);
  v8 = v5 + 1;
  if ( v7 > 0x17u )
  {
    if ( v7 == 86 || v7 == 77 )
      return 0;
    if ( v7 != 53 )
      goto LABEL_6;
    v15 = a4;
    if ( (unsigned __int8)sub_15F8F00(a2) )
      return 0;
    a4 = v15;
    v7 = *(_BYTE *)(a2 + 16);
    if ( v7 > 0x17u )
    {
LABEL_6:
      if ( v7 != 56 )
        return sub_14A4C90((__int64)v8, a2);
      goto LABEL_12;
    }
  }
  if ( v7 != 5 || *(_WORD *)(a2 + 18) != 32 )
    return sub_14A4C90((__int64)v8, a2);
LABEL_12:
  v10 = a3 + 8;
  v11 = a4 - 1;
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v12 = *(__int64 ***)(a2 - 8);
  else
    v12 = (__int64 **)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  v13 = *v12;
  v14 = sub_16348C0(a2);
  return sub_14A1310(v8, v14, v13, v10, v11);
}
