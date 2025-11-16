// Function: sub_2477670
// Address: 0x2477670
//
void __fastcall sub_2477670(__int64 *a1, __int64 a2, char a3)
{
  __int64 *v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  _QWORD *v8; // rax
  unsigned __int64 v9; // [rsp+8h] [rbp-B8h]
  unsigned int *v10[2]; // [rsp+10h] [rbp-B0h] BYREF
  char v11; // [rsp+20h] [rbp-A0h] BYREF
  void *v12; // [rsp+90h] [rbp-30h]

  sub_23D0AB0((__int64)v10, a2, 0, 0, 0);
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v4 = *(__int64 **)(a2 - 8);
  else
    v4 = (__int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v5 = sub_246F3F0((__int64)a1, *v4);
  v6 = sub_B34870((__int64)v10, v5);
  v7 = v6;
  if ( a3 )
  {
    v9 = v6;
    v8 = sub_2463540(a1, *(_QWORD *)(a2 + 8));
    v7 = sub_2464970(a1, v10, v9, (__int64)v8, 0);
  }
  sub_246EF60((__int64)a1, a2, v7);
  if ( *(_DWORD *)(a1[1] + 4) )
    sub_2477350((__int64)a1, a2);
  nullsub_61();
  v12 = &unk_49DA100;
  nullsub_63();
  if ( (char *)v10[0] != &v11 )
    _libc_free((unsigned __int64)v10[0]);
}
