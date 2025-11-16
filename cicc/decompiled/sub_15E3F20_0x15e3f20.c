// Function: sub_15E3F20
// Address: 0x15e3f20
//
__int16 __fastcall sub_15E3F20(__int64 a1, __int64 a2)
{
  _QWORD *v2; // rax
  __int64 v3; // rcx
  unsigned __int64 v4; // rdx
  __int64 v5; // rcx
  _QWORD *v7; // r13
  __int64 v8; // rax
  __int64 **v9; // rax
  __int64 v10; // rax
  __int64 v11; // rcx
  unsigned __int64 v12; // rdx
  __int64 v13; // rdx
  __int64 v14; // rdx
  _QWORD *v15; // r13

  if ( a2 )
  {
    sub_15E3980(a1);
    if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
      v2 = *(_QWORD **)(a1 - 8);
    else
      v2 = (_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
    if ( v2[3] )
    {
      v3 = v2[4];
      v4 = v2[5] & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v4 = v3;
      if ( v3 )
        *(_QWORD *)(v3 + 16) = *(_QWORD *)(v3 + 16) & 3LL | v4;
    }
    v2[3] = a2;
    v5 = *(_QWORD *)(a2 + 8);
    v2[4] = v5;
    if ( v5 )
      *(_QWORD *)(v5 + 16) = (unsigned __int64)(v2 + 4) | *(_QWORD *)(v5 + 16) & 3LL;
    v2[5] = v2[5] & 3LL | (a2 + 8);
    *(_QWORD *)(a2 + 8) = v2 + 3;
  }
  else if ( (*(_DWORD *)(a1 + 20) & 0xFFFFFFF) != 0 )
  {
    if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
      v7 = *(_QWORD **)(a1 - 8);
    else
      v7 = (_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
    v8 = sub_15E0530(a1);
    v9 = (__int64 **)sub_16471A0(v8, 0);
    v10 = sub_1599A20(v9);
    if ( v7[3] )
    {
      v11 = v7[4];
      v12 = v7[5] & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v12 = v11;
      if ( v11 )
        *(_QWORD *)(v11 + 16) = *(_QWORD *)(v11 + 16) & 3LL | v12;
    }
    v7[3] = v10;
    if ( v10 )
    {
      v13 = *(_QWORD *)(v10 + 8);
      v7[4] = v13;
      if ( v13 )
        *(_QWORD *)(v13 + 16) = (unsigned __int64)(v7 + 4) | *(_QWORD *)(v13 + 16) & 3LL;
      v14 = v7[5];
      v15 = v7 + 3;
      v15[2] = (v10 + 8) | v14 & 3;
      *(_QWORD *)(v10 + 8) = v15;
    }
  }
  return sub_15E3BA0(a1, 1, a2 != 0);
}
