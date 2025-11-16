// Function: sub_15FB300
// Address: 0x15fb300
//
void __fastcall sub_15FB300(__int64 a1, int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  __int64 v10; // rdx
  unsigned __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rdx
  unsigned __int64 v15; // rax
  __int64 v16; // rax

  sub_15F1EA0(a1, a5, a2, a1 - 48, 2, a7);
  if ( *(_QWORD *)(a1 - 48) )
  {
    v10 = *(_QWORD *)(a1 - 40);
    v11 = *(_QWORD *)(a1 - 32) & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v11 = v10;
    if ( v10 )
      *(_QWORD *)(v10 + 16) = *(_QWORD *)(v10 + 16) & 3LL | v11;
  }
  *(_QWORD *)(a1 - 48) = a3;
  if ( a3 )
  {
    v12 = *(_QWORD *)(a3 + 8);
    *(_QWORD *)(a1 - 40) = v12;
    if ( v12 )
      *(_QWORD *)(v12 + 16) = (a1 - 40) | *(_QWORD *)(v12 + 16) & 3LL;
    v13 = *(_QWORD *)(a1 - 32);
    *(_QWORD *)(a3 + 8) = a1 - 48;
    *(_QWORD *)(a1 - 32) = (a3 + 8) | v13 & 3;
  }
  if ( *(_QWORD *)(a1 - 24) )
  {
    v14 = *(_QWORD *)(a1 - 16);
    v15 = *(_QWORD *)(a1 - 8) & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v15 = v14;
    if ( v14 )
      *(_QWORD *)(v14 + 16) = *(_QWORD *)(v14 + 16) & 3LL | v15;
  }
  *(_QWORD *)(a1 - 24) = a4;
  if ( a4 )
  {
    v16 = *(_QWORD *)(a4 + 8);
    *(_QWORD *)(a1 - 16) = v16;
    if ( v16 )
      *(_QWORD *)(v16 + 16) = (a1 - 16) | *(_QWORD *)(v16 + 16) & 3LL;
    *(_QWORD *)(a1 - 8) = (a4 + 8) | *(_QWORD *)(a1 - 8) & 3LL;
    *(_QWORD *)(a4 + 8) = a1 - 24;
  }
  sub_164B780(a1, a6);
  nullsub_558();
}
