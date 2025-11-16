// Function: sub_15F8800
// Address: 0x15f8800
//
__int64 __fastcall sub_15F8800(__int64 a1, __int64 a2)
{
  unsigned int v2; // r13d
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rcx
  unsigned __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 result; // rax
  __int64 v10; // rax
  __int64 v11; // rcx
  unsigned __int64 v12; // rdx
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rcx
  unsigned __int64 v16; // rdx
  __int64 v17; // rdx

  v2 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  v3 = sub_16498A0(a2);
  v4 = sub_1643270(v3);
  sub_15F1EA0(a1, v4, 2, a1 - 24LL * v2, v2, 0);
  v5 = *(_QWORD *)(a2 - 24);
  if ( *(_QWORD *)(a1 - 24) )
  {
    v6 = *(_QWORD *)(a1 - 16);
    v7 = *(_QWORD *)(a1 - 8) & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v7 = v6;
    if ( v6 )
      *(_QWORD *)(v6 + 16) = *(_QWORD *)(v6 + 16) & 3LL | v7;
  }
  *(_QWORD *)(a1 - 24) = v5;
  if ( v5 )
  {
    v8 = *(_QWORD *)(v5 + 8);
    *(_QWORD *)(a1 - 16) = v8;
    if ( v8 )
      *(_QWORD *)(v8 + 16) = (a1 - 16) | *(_QWORD *)(v8 + 16) & 3LL;
    *(_QWORD *)(a1 - 8) = (v5 + 8) | *(_QWORD *)(a1 - 8) & 3LL;
    *(_QWORD *)(v5 + 8) = a1 - 24;
  }
  if ( (*(_DWORD *)(a2 + 20) & 0xFFFFFFF) != 1 )
  {
    v10 = *(_QWORD *)(a2 - 72);
    if ( *(_QWORD *)(a1 - 72) )
    {
      v11 = *(_QWORD *)(a1 - 64);
      v12 = *(_QWORD *)(a1 - 56) & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v12 = v11;
      if ( v11 )
        *(_QWORD *)(v11 + 16) = *(_QWORD *)(v11 + 16) & 3LL | v12;
    }
    *(_QWORD *)(a1 - 72) = v10;
    if ( v10 )
    {
      v13 = *(_QWORD *)(v10 + 8);
      *(_QWORD *)(a1 - 64) = v13;
      if ( v13 )
        *(_QWORD *)(v13 + 16) = (a1 - 64) | *(_QWORD *)(v13 + 16) & 3LL;
      *(_QWORD *)(a1 - 56) = (v10 + 8) | *(_QWORD *)(a1 - 56) & 3LL;
      *(_QWORD *)(v10 + 8) = a1 - 72;
    }
    v14 = *(_QWORD *)(a2 - 48);
    if ( *(_QWORD *)(a1 - 48) )
    {
      v15 = *(_QWORD *)(a1 - 40);
      v16 = *(_QWORD *)(a1 - 32) & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v16 = v15;
      if ( v15 )
        *(_QWORD *)(v15 + 16) = *(_QWORD *)(v15 + 16) & 3LL | v16;
    }
    *(_QWORD *)(a1 - 48) = v14;
    if ( v14 )
    {
      v17 = *(_QWORD *)(v14 + 8);
      *(_QWORD *)(a1 - 40) = v17;
      if ( v17 )
        *(_QWORD *)(v17 + 16) = (a1 - 40) | *(_QWORD *)(v17 + 16) & 3LL;
      *(_QWORD *)(a1 - 32) = (v14 + 8) | *(_QWORD *)(a1 - 32) & 3LL;
      *(_QWORD *)(v14 + 8) = a1 - 48;
    }
  }
  result = *(_BYTE *)(a2 + 17) & 0xFE | *(_BYTE *)(a1 + 17) & 1u;
  *(_BYTE *)(a1 + 17) = *(_BYTE *)(a2 + 17) & 0xFE | *(_BYTE *)(a1 + 17) & 1;
  return result;
}
