// Function: sub_B503D0
// Address: 0xb503d0
//
void __fastcall sub_B503D0(__int64 a1, int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7, __int64 a8)
{
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax

  sub_B44260(a1, a5, a2, 2u, a7, a8);
  if ( *(_QWORD *)(a1 - 64) )
  {
    v11 = *(_QWORD *)(a1 - 56);
    **(_QWORD **)(a1 - 48) = v11;
    if ( v11 )
      *(_QWORD *)(v11 + 16) = *(_QWORD *)(a1 - 48);
  }
  *(_QWORD *)(a1 - 64) = a3;
  if ( a3 )
  {
    v12 = *(_QWORD *)(a3 + 16);
    *(_QWORD *)(a1 - 56) = v12;
    if ( v12 )
      *(_QWORD *)(v12 + 16) = a1 - 56;
    *(_QWORD *)(a1 - 48) = a3 + 16;
    *(_QWORD *)(a3 + 16) = a1 - 64;
  }
  if ( *(_QWORD *)(a1 - 32) )
  {
    v13 = *(_QWORD *)(a1 - 24);
    **(_QWORD **)(a1 - 16) = v13;
    if ( v13 )
      *(_QWORD *)(v13 + 16) = *(_QWORD *)(a1 - 16);
  }
  *(_QWORD *)(a1 - 32) = a4;
  if ( a4 )
  {
    v14 = *(_QWORD *)(a4 + 16);
    *(_QWORD *)(a1 - 24) = v14;
    if ( v14 )
      *(_QWORD *)(v14 + 16) = a1 - 24;
    *(_QWORD *)(a1 - 16) = a4 + 16;
    *(_QWORD *)(a4 + 16) = a1 - 32;
  }
  sub_BD6B50(a1, a6);
  nullsub_67();
}
