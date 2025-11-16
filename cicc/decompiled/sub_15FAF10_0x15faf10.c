// Function: sub_15FAF10
// Address: 0x15faf10
//
__int64 __fastcall sub_15FAF10(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rsi
  unsigned __int64 v6; // rdx
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rcx
  unsigned __int64 v11; // rdx
  __int64 v12; // rdx
  __int64 result; // rax
  __int64 v14; // rdx
  unsigned __int64 v15; // rax
  __int64 v16; // rdx
  unsigned __int64 v17; // rax

  sub_15F1EA0(a1, *(_QWORD *)a2, 63, a1 - 48, 2, 0);
  *(_QWORD *)(a1 + 56) = a1 + 72;
  *(_QWORD *)(a1 + 64) = 0x400000000LL;
  if ( *(_DWORD *)(a2 + 64) )
    sub_15F4C60(a1 + 56, a2 + 56);
  v3 = *(_QWORD *)(a2 - 48);
  v4 = *(_QWORD *)(a1 - 48);
  if ( v3 )
  {
    if ( v4 )
    {
      v5 = *(_QWORD *)(a1 - 40);
      v6 = *(_QWORD *)(a1 - 32) & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v6 = v5;
      if ( v5 )
        *(_QWORD *)(v5 + 16) = *(_QWORD *)(v5 + 16) & 3LL | v6;
    }
    *(_QWORD *)(a1 - 48) = v3;
    v7 = *(_QWORD *)(v3 + 8);
    *(_QWORD *)(a1 - 40) = v7;
    if ( v7 )
      *(_QWORD *)(v7 + 16) = (a1 - 40) | *(_QWORD *)(v7 + 16) & 3LL;
    *(_QWORD *)(a1 - 32) = (v3 + 8) | *(_QWORD *)(a1 - 32) & 3LL;
    *(_QWORD *)(v3 + 8) = a1 - 48;
  }
  else if ( v4 )
  {
    v16 = *(_QWORD *)(a1 - 40);
    v17 = *(_QWORD *)(a1 - 32) & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v17 = v16;
    if ( v16 )
      *(_QWORD *)(v16 + 16) = *(_QWORD *)(v16 + 16) & 3LL | v17;
    *(_QWORD *)(a1 - 48) = 0;
  }
  v8 = *(_QWORD *)(a2 - 24);
  v9 = *(_QWORD *)(a1 - 24);
  if ( v8 )
  {
    if ( v9 )
    {
      v10 = *(_QWORD *)(a1 - 16);
      v11 = *(_QWORD *)(a1 - 8) & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v11 = v10;
      if ( v10 )
        *(_QWORD *)(v10 + 16) = *(_QWORD *)(v10 + 16) & 3LL | v11;
    }
    *(_QWORD *)(a1 - 24) = v8;
    v12 = *(_QWORD *)(v8 + 8);
    *(_QWORD *)(a1 - 16) = v12;
    if ( v12 )
      *(_QWORD *)(v12 + 16) = (a1 - 16) | *(_QWORD *)(v12 + 16) & 3LL;
    *(_QWORD *)(a1 - 8) = (v8 + 8) | *(_QWORD *)(a1 - 8) & 3LL;
    *(_QWORD *)(v8 + 8) = a1 - 24;
  }
  else if ( v9 )
  {
    v14 = *(_QWORD *)(a1 - 16);
    v15 = *(_QWORD *)(a1 - 8) & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v15 = v14;
    if ( v14 )
      *(_QWORD *)(v14 + 16) = *(_QWORD *)(v14 + 16) & 3LL | v15;
    *(_QWORD *)(a1 - 24) = 0;
  }
  result = *(_BYTE *)(a2 + 17) & 0xFE | *(_BYTE *)(a1 + 17) & 1u;
  *(_BYTE *)(a1 + 17) = *(_BYTE *)(a2 + 17) & 0xFE | *(_BYTE *)(a1 + 17) & 1;
  return result;
}
