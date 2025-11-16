// Function: sub_14EDD70
// Address: 0x14edd70
//
__int64 __fastcall sub_14EDD70(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rax
  __int64 v9; // r12
  __int64 v10; // r8
  __int64 v11; // rcx
  unsigned __int64 v12; // rdx
  __int64 v13; // rdx
  __int64 v14; // rdx
  unsigned __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rdx
  unsigned __int64 v18; // rax
  __int64 v19; // rax
  __int64 v22; // [rsp+0h] [rbp-40h]

  v8 = sub_1648A60(56, 3);
  v9 = v8;
  if ( v8 )
  {
    v10 = a5;
    v22 = v8 - 72;
    sub_15F1EA0(v8, *a2, 55, v8 - 72, 3, v10);
    if ( *(_QWORD *)(v9 - 72) )
    {
      v11 = *(_QWORD *)(v9 - 64);
      v12 = *(_QWORD *)(v9 - 56) & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v12 = v11;
      if ( v11 )
        *(_QWORD *)(v11 + 16) = *(_QWORD *)(v11 + 16) & 3LL | v12;
    }
    *(_QWORD *)(v9 - 72) = a1;
    if ( a1 )
    {
      v13 = *(_QWORD *)(a1 + 8);
      *(_QWORD *)(v9 - 64) = v13;
      if ( v13 )
        *(_QWORD *)(v13 + 16) = (v9 - 64) | *(_QWORD *)(v13 + 16) & 3LL;
      *(_QWORD *)(v9 - 56) = (a1 + 8) | *(_QWORD *)(v9 - 56) & 3LL;
      *(_QWORD *)(a1 + 8) = v22;
    }
    if ( *(_QWORD *)(v9 - 48) )
    {
      v14 = *(_QWORD *)(v9 - 40);
      v15 = *(_QWORD *)(v9 - 32) & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v15 = v14;
      if ( v14 )
        *(_QWORD *)(v14 + 16) = *(_QWORD *)(v14 + 16) & 3LL | v15;
    }
    *(_QWORD *)(v9 - 48) = a2;
    v16 = a2[1];
    *(_QWORD *)(v9 - 40) = v16;
    if ( v16 )
      *(_QWORD *)(v16 + 16) = (v9 - 40) | *(_QWORD *)(v16 + 16) & 3LL;
    *(_QWORD *)(v9 - 32) = (unsigned __int64)(a2 + 1) | *(_QWORD *)(v9 - 32) & 3LL;
    a2[1] = v9 - 48;
    if ( *(_QWORD *)(v9 - 24) )
    {
      v17 = *(_QWORD *)(v9 - 16);
      v18 = *(_QWORD *)(v9 - 8) & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v18 = v17;
      if ( v17 )
        *(_QWORD *)(v17 + 16) = *(_QWORD *)(v17 + 16) & 3LL | v18;
    }
    *(_QWORD *)(v9 - 24) = a3;
    if ( a3 )
    {
      v19 = *(_QWORD *)(a3 + 8);
      *(_QWORD *)(v9 - 16) = v19;
      if ( v19 )
        *(_QWORD *)(v19 + 16) = (v9 - 16) | *(_QWORD *)(v19 + 16) & 3LL;
      *(_QWORD *)(v9 - 8) = (a3 + 8) | *(_QWORD *)(v9 - 8) & 3LL;
      *(_QWORD *)(a3 + 8) = v9 - 24;
    }
    sub_164B780(v9, a4);
  }
  if ( a6 )
    sub_15F4370(v9, a6, 0, 0);
  return v9;
}
