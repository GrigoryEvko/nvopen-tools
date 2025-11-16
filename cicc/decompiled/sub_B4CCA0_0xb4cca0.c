// Function: sub_B4CCA0
// Address: 0xb4cca0
//
__int64 __fastcall sub_B4CCA0(
        __int64 a1,
        __int64 *a2,
        unsigned int a3,
        __int64 a4,
        unsigned __int8 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8)
{
  unsigned int v8; // r10d
  __int16 v10; // r13
  __int64 v12; // rbx
  __int64 v13; // r8
  unsigned __int16 v14; // r9
  __int64 v15; // rdi
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  __int16 v19; // r8
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // [rsp+10h] [rbp-40h]
  unsigned __int16 v25; // [rsp+18h] [rbp-38h]

  v8 = a3;
  v10 = a5;
  v12 = a4;
  v13 = a7;
  v14 = a8;
  v15 = *a2;
  if ( !a4 )
  {
    v21 = sub_BCB2D0(v15);
    v22 = sub_ACD640(v21, 1, 0);
    v15 = *a2;
    v13 = a7;
    v14 = a8;
    v8 = a3;
    v12 = v22;
  }
  v23 = v13;
  v25 = v14;
  v16 = sub_BCE3C0(v15, v8);
  sub_B44260(a1, v16, 31, 1u, v23, v25);
  if ( *(_QWORD *)(a1 - 32) )
  {
    v17 = *(_QWORD *)(a1 - 24);
    **(_QWORD **)(a1 - 16) = v17;
    if ( v17 )
      *(_QWORD *)(v17 + 16) = *(_QWORD *)(a1 - 16);
  }
  *(_QWORD *)(a1 - 32) = v12;
  if ( v12 )
  {
    v18 = *(_QWORD *)(v12 + 16);
    *(_QWORD *)(a1 - 24) = v18;
    if ( v18 )
      *(_QWORD *)(v18 + 16) = a1 - 24;
    *(_QWORD *)(a1 - 16) = v12 + 16;
    *(_QWORD *)(v12 + 16) = a1 - 32;
  }
  v19 = *(_WORD *)(a1 + 2);
  *(_QWORD *)(a1 + 72) = a2;
  *(_WORD *)(a1 + 2) = v10 | v19 & 0xFFC0;
  return sub_BD6B50(a1, a6);
}
