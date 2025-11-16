// Function: sub_15F9100
// Address: 0x15f9100
//
void __fastcall sub_15F9100(__int64 a1, _QWORD *a2, _BYTE *a3, __int64 a4)
{
  __int64 v5; // rdx
  unsigned __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rax
  _BYTE *v9; // [rsp+0h] [rbp-40h] BYREF
  char v10; // [rsp+10h] [rbp-30h]
  char v11; // [rsp+11h] [rbp-2Fh]

  sub_15F1EA0(a1, *(_QWORD *)(*a2 + 24LL), 30, a1 - 24, 1, a4);
  if ( *(_QWORD *)(a1 - 24) )
  {
    v5 = *(_QWORD *)(a1 - 16);
    v6 = *(_QWORD *)(a1 - 8) & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v6 = v5;
    if ( v5 )
      *(_QWORD *)(v5 + 16) = *(_QWORD *)(v5 + 16) & 3LL | v6;
  }
  v7 = a2[1];
  *(_QWORD *)(a1 - 24) = a2;
  *(_QWORD *)(a1 - 16) = v7;
  if ( v7 )
    *(_QWORD *)(v7 + 16) = (a1 - 16) | *(_QWORD *)(v7 + 16) & 3LL;
  v8 = *(_QWORD *)(a1 - 8);
  a2[1] = a1 - 24;
  *(_WORD *)(a1 + 18) &= ~1u;
  *(_QWORD *)(a1 - 8) = (unsigned __int64)(a2 + 1) | v8 & 3;
  sub_15F8F50(a1, 0);
  *(_WORD *)(a1 + 18) &= 0xFC7Fu;
  *(_BYTE *)(a1 + 56) = 1;
  nullsub_556();
  if ( a3 )
  {
    if ( *a3 )
    {
      v9 = a3;
      v11 = 1;
      v10 = 3;
      sub_164B780(a1, &v9);
    }
  }
}
