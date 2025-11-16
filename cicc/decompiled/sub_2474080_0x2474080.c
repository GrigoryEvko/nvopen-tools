// Function: sub_2474080
// Address: 0x2474080
//
void __fastcall sub_2474080(__int64 a1, __int64 a2)
{
  __int64 *v3; // rax
  __int64 v4; // rbx
  __int64 v5; // r15
  _QWORD *v6; // rcx
  _QWORD *v7; // rax
  __int64 v8; // rsi
  __int64 v9; // rsi
  _QWORD *v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // [rsp+8h] [rbp-C8h]
  unsigned int *v14[2]; // [rsp+10h] [rbp-C0h] BYREF
  char v15; // [rsp+20h] [rbp-B0h] BYREF
  void *v16; // [rsp+90h] [rbp-40h]

  sub_23D0AB0((__int64)v14, a2, 0, 0, 0);
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v3 = *(__int64 **)(a2 - 8);
  else
    v3 = (__int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v4 = v3[4];
  v5 = *v3;
  v6 = sub_2463540((__int64 *)a1, *(_QWORD *)(v4 + 8));
  if ( **(_BYTE **)(a1 + 8) )
  {
    v13 = (__int64)sub_2465B30((__int64 *)a1, v5, (__int64)v14, (__int64)v6, 1);
    if ( !(_BYTE)qword_4FE84C8 )
      goto LABEL_5;
LABEL_14:
    sub_2472230(a1, v5, a2);
    if ( *(_BYTE *)a2 != 65 )
      goto LABEL_6;
    goto LABEL_15;
  }
  v13 = sub_2463FC0(a1, v5, v14, 0x100u);
  if ( (_BYTE)qword_4FE84C8 )
    goto LABEL_14;
LABEL_5:
  if ( *(_BYTE *)a2 != 65 )
    goto LABEL_6;
LABEL_15:
  sub_2472230(a1, v4, a2);
LABEL_6:
  v7 = sub_2463540((__int64 *)a1, *(_QWORD *)(v4 + 8));
  v8 = (__int64)v7;
  if ( v7 )
    v8 = sub_AD6530((__int64)v7, (__int64)v7);
  sub_2463EC0((__int64 *)v14, v8, v13, 0, 0);
  v9 = *(_QWORD *)(a2 + 8);
  v10 = sub_2463540((__int64 *)a1, v9);
  v11 = (__int64)v10;
  if ( v10 )
    v11 = sub_AD6530((__int64)v10, v9);
  sub_246EF60(a1, a2, v11);
  v12 = sub_AD6530(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 88LL), a2);
  sub_246F1C0(a1, a2, v12);
  nullsub_61();
  v16 = &unk_49DA100;
  nullsub_63();
  if ( (char *)v14[0] != &v15 )
    _libc_free((unsigned __int64)v14[0]);
}
