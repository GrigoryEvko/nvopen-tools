// Function: sub_15E3980
// Address: 0x15e3980
//
void __fastcall sub_15E3980(__int64 a1)
{
  __int64 v1; // rax
  __int64 **v2; // rax
  __int64 v3; // rax
  __int64 *v4; // rdx
  __int64 v5; // rsi
  unsigned __int64 v6; // rcx
  __int64 v7; // rcx
  _QWORD *v8; // rdx
  __int64 v9; // rsi
  unsigned __int64 v10; // rcx
  __int64 v11; // rcx
  __int64 v12; // rcx
  _QWORD *v13; // rdx
  _QWORD *v14; // rbx
  __int64 v15; // rcx
  unsigned __int64 v16; // rdx
  __int64 v17; // rdx
  __int64 v18; // rdx
  _QWORD *v19; // rbx

  if ( (*(_DWORD *)(a1 + 20) & 0xFFFFFFF) == 0 )
  {
    sub_1648880(a1, 3, 0);
    *(_DWORD *)(a1 + 20) = *(_DWORD *)(a1 + 20) & 0xF0000000 | 3;
    v1 = sub_15E0530(a1);
    v2 = (__int64 **)sub_16471A0(v1, 0);
    v3 = sub_1599A20(v2);
    if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
      v4 = *(__int64 **)(a1 - 8);
    else
      v4 = (__int64 *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
    if ( *v4 )
    {
      v5 = v4[1];
      v6 = v4[2] & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v6 = v5;
      if ( v5 )
        *(_QWORD *)(v5 + 16) = *(_QWORD *)(v5 + 16) & 3LL | v6;
    }
    *v4 = v3;
    if ( v3 )
    {
      v7 = *(_QWORD *)(v3 + 8);
      v4[1] = v7;
      if ( v7 )
        *(_QWORD *)(v7 + 16) = (unsigned __int64)(v4 + 1) | *(_QWORD *)(v7 + 16) & 3LL;
      v4[2] = (v3 + 8) | v4[2] & 3;
      *(_QWORD *)(v3 + 8) = v4;
    }
    if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
      v8 = *(_QWORD **)(a1 - 8);
    else
      v8 = (_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
    if ( v8[3] )
    {
      v9 = v8[4];
      v10 = v8[5] & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v10 = v9;
      if ( v9 )
        *(_QWORD *)(v9 + 16) = *(_QWORD *)(v9 + 16) & 3LL | v10;
    }
    v8[3] = v3;
    if ( v3 )
    {
      v11 = *(_QWORD *)(v3 + 8);
      v8[4] = v11;
      if ( v11 )
        *(_QWORD *)(v11 + 16) = (unsigned __int64)(v8 + 4) | *(_QWORD *)(v11 + 16) & 3LL;
      v12 = v8[5];
      v13 = v8 + 3;
      v13[2] = (v3 + 8) | v12 & 3;
      *(_QWORD *)(v3 + 8) = v13;
    }
    if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
      v14 = *(_QWORD **)(a1 - 8);
    else
      v14 = (_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
    if ( v14[6] )
    {
      v15 = v14[7];
      v16 = v14[8] & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v16 = v15;
      if ( v15 )
        *(_QWORD *)(v15 + 16) = *(_QWORD *)(v15 + 16) & 3LL | v16;
    }
    v14[6] = v3;
    if ( v3 )
    {
      v17 = *(_QWORD *)(v3 + 8);
      v14[7] = v17;
      if ( v17 )
        *(_QWORD *)(v17 + 16) = (unsigned __int64)(v14 + 7) | *(_QWORD *)(v17 + 16) & 3LL;
      v18 = v14[8];
      v19 = v14 + 6;
      v19[2] = (v3 + 8) | v18 & 3;
      *(_QWORD *)(v3 + 8) = v19;
    }
  }
}
