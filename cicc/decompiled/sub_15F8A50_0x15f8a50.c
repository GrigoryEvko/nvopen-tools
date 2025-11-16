// Function: sub_15F8A50
// Address: 0x15f8a50
//
__int64 __fastcall sub_15F8A50(
        __int64 a1,
        _QWORD *a2,
        unsigned int a3,
        __int64 a4,
        unsigned int a5,
        __int64 a6,
        __int64 a7)
{
  __int64 v7; // r15
  __int64 v9; // rbx
  __int64 v10; // rax
  unsigned int v11; // r10d
  __int64 v12; // rdx
  unsigned __int64 v13; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax

  v7 = a1 - 24;
  if ( !a4 )
  {
    v16 = sub_1643350(*a2);
    v9 = sub_159C470(v16, 1, 0);
    v17 = sub_1646BA0(a2, a3);
    sub_15F1EA0(a1, v17, 29, v7, 1, a7);
    v11 = a5;
    if ( !*(_QWORD *)(a1 - 24) )
      goto LABEL_5;
    goto LABEL_3;
  }
  v9 = a4;
  v10 = sub_1646BA0(a2, a3);
  sub_15F1EA0(a1, v10, 29, v7, 1, a7);
  v11 = a5;
  if ( *(_QWORD *)(a1 - 24) )
  {
LABEL_3:
    v12 = *(_QWORD *)(a1 - 16);
    v13 = *(_QWORD *)(a1 - 8) & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v13 = v12;
    if ( v12 )
      *(_QWORD *)(v12 + 16) = *(_QWORD *)(v12 + 16) & 3LL | v13;
LABEL_5:
    *(_QWORD *)(a1 - 24) = v9;
    if ( !v9 )
      goto LABEL_6;
    goto LABEL_8;
  }
  *(_QWORD *)(a1 - 24) = v9;
LABEL_8:
  v15 = *(_QWORD *)(v9 + 8);
  *(_QWORD *)(a1 - 16) = v15;
  if ( v15 )
    *(_QWORD *)(v15 + 16) = (a1 - 16) | *(_QWORD *)(v15 + 16) & 3LL;
  *(_QWORD *)(a1 - 8) = (v9 + 8) | *(_QWORD *)(a1 - 8) & 3LL;
  *(_QWORD *)(v9 + 8) = v7;
LABEL_6:
  *(_QWORD *)(a1 + 56) = a2;
  sub_15F8A20(a1, v11);
  return sub_164B780(a1, a6);
}
