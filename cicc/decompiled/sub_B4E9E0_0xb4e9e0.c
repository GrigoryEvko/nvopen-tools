// Function: sub_B4E9E0
// Address: 0xb4e9e0
//
__int64 __fastcall sub_B4E9E0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        void *a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8)
{
  __int64 v11; // rax
  __int64 v12; // rax
  bool v13; // zf
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v20; // [rsp+28h] [rbp-38h]

  v11 = *(_QWORD *)(a2 + 8);
  LODWORD(v20) = a5;
  BYTE4(v20) = *(_BYTE *)(v11 + 8) == 18;
  v12 = sub_BCE1B0(*(_QWORD *)(v11 + 24), v20);
  sub_B44260(a1, v12, 63, 2u, a7, a8);
  v13 = *(_QWORD *)(a1 - 64) == 0;
  *(_QWORD *)(a1 + 72) = a1 + 88;
  *(_QWORD *)(a1 + 80) = 0x400000000LL;
  if ( !v13 )
  {
    v14 = *(_QWORD *)(a1 - 56);
    **(_QWORD **)(a1 - 48) = v14;
    if ( v14 )
      *(_QWORD *)(v14 + 16) = *(_QWORD *)(a1 - 48);
  }
  v15 = *(_QWORD *)(a2 + 16);
  *(_QWORD *)(a1 - 64) = a2;
  *(_QWORD *)(a1 - 56) = v15;
  if ( v15 )
    *(_QWORD *)(v15 + 16) = a1 - 56;
  v13 = *(_QWORD *)(a1 - 32) == 0;
  *(_QWORD *)(a1 - 48) = a2 + 16;
  *(_QWORD *)(a2 + 16) = a1 - 64;
  if ( !v13 )
  {
    v16 = *(_QWORD *)(a1 - 24);
    **(_QWORD **)(a1 - 16) = v16;
    if ( v16 )
      *(_QWORD *)(v16 + 16) = *(_QWORD *)(a1 - 16);
  }
  *(_QWORD *)(a1 - 32) = a3;
  if ( a3 )
  {
    v17 = *(_QWORD *)(a3 + 16);
    *(_QWORD *)(a1 - 24) = v17;
    if ( v17 )
      *(_QWORD *)(v17 + 16) = a1 - 24;
    *(_QWORD *)(a1 - 16) = a3 + 16;
    *(_QWORD *)(a3 + 16) = a1 - 32;
  }
  sub_B4E7F0(a1, a4, a5);
  return sub_BD6B50(a1, a6);
}
