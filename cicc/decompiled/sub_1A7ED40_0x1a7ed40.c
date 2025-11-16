// Function: sub_1A7ED40
// Address: 0x1a7ed40
//
__int64 __fastcall sub_1A7ED40(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r12
  __int64 v5; // rdx
  _QWORD *v6; // rax
  __int64 v7; // rcx
  unsigned __int64 v8; // rdx
  __int64 v9; // rcx
  _QWORD v11[2]; // [rsp+0h] [rbp-50h] BYREF
  _QWORD *v12; // [rsp+10h] [rbp-40h] BYREF
  __int16 v13; // [rsp+20h] [rbp-30h]

  v4 = sub_15F4880(a1);
  v11[0] = sub_1649960(a1);
  v13 = 261;
  v11[1] = v5;
  v12 = v11;
  sub_164B780(v4, (__int64 *)&v12);
  sub_15F2120(v4, a2);
  if ( a3 )
  {
    if ( (*(_BYTE *)(v4 + 23) & 0x40) != 0 )
      v6 = *(_QWORD **)(v4 - 8);
    else
      v6 = (_QWORD *)(v4 - 24LL * (*(_DWORD *)(v4 + 20) & 0xFFFFFFF));
    if ( *v6 )
    {
      v7 = v6[1];
      v8 = v6[2] & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v8 = v7;
      if ( v7 )
        *(_QWORD *)(v7 + 16) = *(_QWORD *)(v7 + 16) & 3LL | v8;
    }
    *v6 = a3;
    v9 = *(_QWORD *)(a3 + 8);
    v6[1] = v9;
    if ( v9 )
      *(_QWORD *)(v9 + 16) = (unsigned __int64)(v6 + 1) | *(_QWORD *)(v9 + 16) & 3LL;
    v6[2] = v6[2] & 3LL | (a3 + 8);
    *(_QWORD *)(a3 + 8) = v6;
  }
  return v4;
}
