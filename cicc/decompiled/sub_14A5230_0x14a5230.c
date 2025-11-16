// Function: sub_14A5230
// Address: 0x14a5230
//
__int64 __fastcall sub_14A5230(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 *v4; // r13
  unsigned __int8 v5; // al
  __int64 v7; // r12
  __int64 v8; // rbx
  __int64 **v9; // rax
  __int64 *v10; // r14
  __int64 v11; // rax
  __int64 v12; // [rsp+8h] [rbp-38h]
  __int64 v13; // [rsp+10h] [rbp-30h]

  v4 = (__int64 *)(a1 + 8);
  v5 = *(_BYTE *)(a2 + 16);
  if ( v5 > 0x17u )
  {
    if ( v5 != 86 && v5 != 77 )
    {
      if ( v5 != 53 )
        goto LABEL_8;
      v12 = a4;
      v13 = a3;
      if ( !(unsigned __int8)sub_15F8F00(a2) )
      {
        a3 = v13;
        a4 = v12;
        v5 = *(_BYTE *)(a2 + 16);
        if ( v5 <= 0x17u )
          goto LABEL_2;
LABEL_8:
        if ( v5 != 56 )
          return sub_14A4C90((__int64)v4, a2);
        goto LABEL_9;
      }
    }
    return 0;
  }
LABEL_2:
  if ( v5 != 5 || *(_WORD *)(a2 + 18) != 32 )
    return sub_14A4C90((__int64)v4, a2);
LABEL_9:
  v7 = a3 + 8;
  v8 = a4 - 1;
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v9 = *(__int64 ***)(a2 - 8);
  else
    v9 = (__int64 **)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  v10 = *v9;
  v11 = sub_16348C0(a2);
  return sub_14A1310(v4, v11, v10, v7, v8);
}
