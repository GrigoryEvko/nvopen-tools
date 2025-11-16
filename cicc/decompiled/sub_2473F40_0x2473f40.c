// Function: sub_2473F40
// Address: 0x2473f40
//
void __fastcall sub_2473F40(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 *v3; // rdx
  __int64 v4; // rax
  __int64 *v5; // rdx
  __int64 v6; // rax
  char *v7; // [rsp+0h] [rbp-A0h] BYREF
  char v8; // [rsp+10h] [rbp-90h] BYREF
  void *v9; // [rsp+80h] [rbp-20h]

  sub_23D0AB0((__int64)&v7, a2, 0, 0, 0);
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v2 = *(_QWORD *)(a2 - 8);
  else
    v2 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  sub_2472230(a1, *(_QWORD *)(v2 + 32), a2);
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v3 = *(__int64 **)(a2 - 8);
  else
    v3 = (__int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v4 = sub_246F3F0(a1, *v3);
  sub_246EF60(a1, a2, v4);
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v5 = *(__int64 **)(a2 - 8);
  else
    v5 = (__int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v6 = sub_246EE10(a1, *v5);
  sub_246F1C0(a1, a2, v6);
  nullsub_61();
  v9 = &unk_49DA100;
  nullsub_63();
  if ( v7 != &v8 )
    _libc_free((unsigned __int64)v7);
}
