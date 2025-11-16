// Function: sub_15FA660
// Address: 0x15fa660
//
__int64 __fastcall sub_15FA660(__int64 a1, _QWORD *a2, __int64 a3, _QWORD *a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rax
  __int64 v9; // rdx
  unsigned __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  bool v13; // zf
  __int64 v14; // rdx
  unsigned __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rdx
  unsigned __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rax

  v8 = sub_16463B0(*(_QWORD *)(*a2 + 24LL), *(_QWORD *)(*a4 + 32LL));
  sub_15F1EA0(a1, v8, 61, a1 - 72, 3, a6);
  if ( *(_QWORD *)(a1 - 72) )
  {
    v9 = *(_QWORD *)(a1 - 64);
    v10 = *(_QWORD *)(a1 - 56) & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v10 = v9;
    if ( v9 )
      *(_QWORD *)(v9 + 16) = *(_QWORD *)(v9 + 16) & 3LL | v10;
  }
  v11 = a2[1];
  *(_QWORD *)(a1 - 72) = a2;
  *(_QWORD *)(a1 - 64) = v11;
  if ( v11 )
    *(_QWORD *)(v11 + 16) = (a1 - 64) | *(_QWORD *)(v11 + 16) & 3LL;
  v12 = *(_QWORD *)(a1 - 56);
  a2[1] = a1 - 72;
  v13 = *(_QWORD *)(a1 - 48) == 0;
  *(_QWORD *)(a1 - 56) = (unsigned __int64)(a2 + 1) | v12 & 3;
  if ( !v13 )
  {
    v14 = *(_QWORD *)(a1 - 40);
    v15 = *(_QWORD *)(a1 - 32) & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v15 = v14;
    if ( v14 )
      *(_QWORD *)(v14 + 16) = *(_QWORD *)(v14 + 16) & 3LL | v15;
  }
  *(_QWORD *)(a1 - 48) = a3;
  if ( a3 )
  {
    v16 = *(_QWORD *)(a3 + 8);
    *(_QWORD *)(a1 - 40) = v16;
    if ( v16 )
      *(_QWORD *)(v16 + 16) = (a1 - 40) | *(_QWORD *)(v16 + 16) & 3LL;
    *(_QWORD *)(a1 - 32) = (a3 + 8) | *(_QWORD *)(a1 - 32) & 3LL;
    *(_QWORD *)(a3 + 8) = a1 - 48;
  }
  if ( *(_QWORD *)(a1 - 24) )
  {
    v17 = *(_QWORD *)(a1 - 16);
    v18 = *(_QWORD *)(a1 - 8) & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v18 = v17;
    if ( v17 )
      *(_QWORD *)(v17 + 16) = *(_QWORD *)(v17 + 16) & 3LL | v18;
  }
  v19 = a4[1];
  *(_QWORD *)(a1 - 24) = a4;
  *(_QWORD *)(a1 - 16) = v19;
  if ( v19 )
    *(_QWORD *)(v19 + 16) = (a1 - 16) | *(_QWORD *)(v19 + 16) & 3LL;
  v20 = *(_QWORD *)(a1 - 8);
  a4[1] = a1 - 24;
  *(_QWORD *)(a1 - 8) = (unsigned __int64)(a4 + 1) | v20 & 3;
  return sub_164B780(a1, a5);
}
