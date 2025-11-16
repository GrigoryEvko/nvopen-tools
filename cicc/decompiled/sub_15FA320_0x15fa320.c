// Function: sub_15FA320
// Address: 0x15fa320
//
__int64 __fastcall sub_15FA320(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v7; // rdx
  unsigned __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax
  bool v11; // zf
  __int64 v12; // rdx
  unsigned __int64 v13; // rax
  __int64 v14; // rax

  sub_15F1EA0(a1, *(_QWORD *)(*a2 + 24LL), 59, a1 - 48, 2, a5);
  if ( *(_QWORD *)(a1 - 48) )
  {
    v7 = *(_QWORD *)(a1 - 40);
    v8 = *(_QWORD *)(a1 - 32) & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v8 = v7;
    if ( v7 )
      *(_QWORD *)(v7 + 16) = *(_QWORD *)(v7 + 16) & 3LL | v8;
  }
  v9 = a2[1];
  *(_QWORD *)(a1 - 48) = a2;
  *(_QWORD *)(a1 - 40) = v9;
  if ( v9 )
    *(_QWORD *)(v9 + 16) = (a1 - 40) | *(_QWORD *)(v9 + 16) & 3LL;
  v10 = *(_QWORD *)(a1 - 32);
  a2[1] = a1 - 48;
  v11 = *(_QWORD *)(a1 - 24) == 0;
  *(_QWORD *)(a1 - 32) = (unsigned __int64)(a2 + 1) | v10 & 3;
  if ( !v11 )
  {
    v12 = *(_QWORD *)(a1 - 16);
    v13 = *(_QWORD *)(a1 - 8) & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v13 = v12;
    if ( v12 )
      *(_QWORD *)(v12 + 16) = *(_QWORD *)(v12 + 16) & 3LL | v13;
  }
  *(_QWORD *)(a1 - 24) = a3;
  if ( a3 )
  {
    v14 = *(_QWORD *)(a3 + 8);
    *(_QWORD *)(a1 - 16) = v14;
    if ( v14 )
      *(_QWORD *)(v14 + 16) = (a1 - 16) | *(_QWORD *)(v14 + 16) & 3LL;
    *(_QWORD *)(a1 - 8) = (a3 + 8) | *(_QWORD *)(a1 - 8) & 3LL;
    *(_QWORD *)(a3 + 8) = a1 - 24;
  }
  return sub_164B780(a1, a4);
}
