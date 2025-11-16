// Function: sub_1601230
// Address: 0x1601230
//
__int64 __fastcall sub_1601230(__int64 a1)
{
  __int64 v1; // r13
  __int64 *v2; // rbx
  __int64 v3; // r14
  __int64 v4; // rax
  __int64 v5; // r12
  __int64 v6; // r15
  __int64 v7; // rdx
  unsigned __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rdx
  unsigned __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rdx
  unsigned __int64 v14; // rax
  __int64 v15; // rax
  char v17[16]; // [rsp+0h] [rbp-50h] BYREF
  __int16 v18; // [rsp+10h] [rbp-40h]

  v1 = *(_QWORD *)(a1 - 24);
  v2 = *(__int64 **)(a1 - 48);
  v18 = 257;
  v3 = *(_QWORD *)(a1 - 72);
  v4 = sub_1648A60(56, 3);
  v5 = v4;
  if ( v4 )
  {
    v6 = v4 - 72;
    sub_15F1EA0(v4, *v2, 55, v4 - 72, 3, 0);
    if ( *(_QWORD *)(v5 - 72) )
    {
      v7 = *(_QWORD *)(v5 - 64);
      v8 = *(_QWORD *)(v5 - 56) & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v8 = v7;
      if ( v7 )
        *(_QWORD *)(v7 + 16) = *(_QWORD *)(v7 + 16) & 3LL | v8;
    }
    *(_QWORD *)(v5 - 72) = v3;
    if ( v3 )
    {
      v9 = *(_QWORD *)(v3 + 8);
      *(_QWORD *)(v5 - 64) = v9;
      if ( v9 )
        *(_QWORD *)(v9 + 16) = (v5 - 64) | *(_QWORD *)(v9 + 16) & 3LL;
      *(_QWORD *)(v5 - 56) = (v3 + 8) | *(_QWORD *)(v5 - 56) & 3LL;
      *(_QWORD *)(v3 + 8) = v6;
    }
    if ( *(_QWORD *)(v5 - 48) )
    {
      v10 = *(_QWORD *)(v5 - 40);
      v11 = *(_QWORD *)(v5 - 32) & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v11 = v10;
      if ( v10 )
        *(_QWORD *)(v10 + 16) = *(_QWORD *)(v10 + 16) & 3LL | v11;
    }
    *(_QWORD *)(v5 - 48) = v2;
    v12 = v2[1];
    *(_QWORD *)(v5 - 40) = v12;
    if ( v12 )
      *(_QWORD *)(v12 + 16) = (v5 - 40) | *(_QWORD *)(v12 + 16) & 3LL;
    *(_QWORD *)(v5 - 32) = (unsigned __int64)(v2 + 1) | *(_QWORD *)(v5 - 32) & 3LL;
    v2[1] = v5 - 48;
    if ( *(_QWORD *)(v5 - 24) )
    {
      v13 = *(_QWORD *)(v5 - 16);
      v14 = *(_QWORD *)(v5 - 8) & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v14 = v13;
      if ( v13 )
        *(_QWORD *)(v13 + 16) = *(_QWORD *)(v13 + 16) & 3LL | v14;
    }
    *(_QWORD *)(v5 - 24) = v1;
    if ( v1 )
    {
      v15 = *(_QWORD *)(v1 + 8);
      *(_QWORD *)(v5 - 16) = v15;
      if ( v15 )
        *(_QWORD *)(v15 + 16) = (v5 - 16) | *(_QWORD *)(v15 + 16) & 3LL;
      *(_QWORD *)(v5 - 8) = (v1 + 8) | *(_QWORD *)(v5 - 8) & 3LL;
      *(_QWORD *)(v1 + 8) = v5 - 24;
    }
    sub_164B780(v5, v17);
  }
  return v5;
}
