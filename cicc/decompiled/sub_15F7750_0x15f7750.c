// Function: sub_15F7750
// Address: 0x15f7750
//
void __fastcall sub_15F7750(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rcx
  unsigned __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rcx
  unsigned __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rdi

  if ( *(_QWORD *)(a1 - 48) )
  {
    v3 = *(_QWORD *)(a1 - 40);
    v4 = *(_QWORD *)(a1 - 32) & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v4 = v3;
    if ( v3 )
      *(_QWORD *)(v3 + 16) = *(_QWORD *)(v3 + 16) & 3LL | v4;
  }
  *(_QWORD *)(a1 - 48) = a2;
  if ( a2 )
  {
    v5 = *(_QWORD *)(a2 + 8);
    *(_QWORD *)(a1 - 40) = v5;
    if ( v5 )
      *(_QWORD *)(v5 + 16) = (a1 - 40) | *(_QWORD *)(v5 + 16) & 3LL;
    *(_QWORD *)(a1 - 32) = (a2 + 8) | *(_QWORD *)(a1 - 32) & 3LL;
    *(_QWORD *)(a2 + 8) = a1 - 48;
  }
  if ( *(_QWORD *)(a1 - 24) )
  {
    v6 = *(_QWORD *)(a1 - 16);
    v7 = *(_QWORD *)(a1 - 8) & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v7 = v6;
    if ( v6 )
      *(_QWORD *)(v6 + 16) = *(_QWORD *)(v6 + 16) & 3LL | v7;
  }
  *(_QWORD *)(a1 - 24) = a3;
  if ( a3 )
  {
    v8 = *(_QWORD *)(a3 + 8);
    *(_QWORD *)(a1 - 16) = v8;
    if ( v8 )
      *(_QWORD *)(v8 + 16) = (a1 - 16) | *(_QWORD *)(v8 + 16) & 3LL;
    v9 = *(_QWORD *)(a1 - 8);
    v10 = a1 - 24;
    *(_QWORD *)(v10 + 16) = (a3 + 8) | v9 & 3;
    *(_QWORD *)(a3 + 8) = v10;
  }
}
