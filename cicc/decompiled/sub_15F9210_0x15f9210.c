// Function: sub_15F9210
// Address: 0x15f9210
//
void __fastcall sub_15F9210(__int64 a1, __int64 a2, __int64 a3, _BYTE *a4, unsigned __int8 a5, __int64 a6)
{
  __int64 v9; // rdx
  unsigned __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  _BYTE *v13; // [rsp+0h] [rbp-50h] BYREF
  char v14; // [rsp+10h] [rbp-40h]
  char v15; // [rsp+11h] [rbp-3Fh]

  sub_15F1EA0(a1, a2, 30, a1 - 24, 1, a6);
  if ( *(_QWORD *)(a1 - 24) )
  {
    v9 = *(_QWORD *)(a1 - 16);
    v10 = *(_QWORD *)(a1 - 8) & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v10 = v9;
    if ( v9 )
      *(_QWORD *)(v9 + 16) = *(_QWORD *)(v9 + 16) & 3LL | v10;
  }
  *(_QWORD *)(a1 - 24) = a3;
  if ( a3 )
  {
    v11 = *(_QWORD *)(a3 + 8);
    *(_QWORD *)(a1 - 16) = v11;
    if ( v11 )
      *(_QWORD *)(v11 + 16) = (a1 - 16) | *(_QWORD *)(v11 + 16) & 3LL;
    v12 = *(_QWORD *)(a1 - 8);
    *(_QWORD *)(a3 + 8) = a1 - 24;
    *(_QWORD *)(a1 - 8) = (a3 + 8) | v12 & 3;
  }
  *(_WORD *)(a1 + 18) = a5 | *(_WORD *)(a1 + 18) & 0xFFFE;
  sub_15F8F50(a1, 0);
  *(_WORD *)(a1 + 18) &= 0xFC7Fu;
  *(_BYTE *)(a1 + 56) = 1;
  nullsub_556();
  if ( a4 )
  {
    if ( *a4 )
    {
      v13 = a4;
      v15 = 1;
      v14 = 3;
      sub_164B780(a1, &v13);
    }
  }
}
