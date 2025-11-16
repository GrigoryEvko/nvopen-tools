// Function: sub_7F7440
// Address: 0x7f7440
//
unsigned __int64 __fastcall sub_7F7440(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  unsigned __int64 v3; // r14
  _BYTE *v4; // r15
  unsigned __int64 v5; // r13
  _BYTE *v6; // rbx
  _BYTE *v7; // rax
  char v8; // al
  char v9; // al
  int v11; // r8d

  v2 = sub_8D4490(a2);
  v3 = *(_QWORD *)(a1 + 184);
  v4 = *(_BYTE **)(a1 + 176);
  v5 = v2;
  v6 = sub_724D50(10);
  *((_QWORD *)v6 + 16) = a2;
  if ( v5 != 1 || (v11 = sub_8D3410(*(_QWORD *)(a2 + 160)), v7 = v4, v11) )
  {
    v7 = sub_724D50(11);
    *((_QWORD *)v7 + 23) = v5;
    *((_QWORD *)v7 + 22) = v4;
    if ( (*(v4 - 8) & 8) == 0 )
      *(v7 - 8) &= ~8u;
  }
  *((_QWORD *)v6 + 22) = v7;
  *((_QWORD *)v6 + 23) = v7;
  v8 = v4[173];
  if ( v8 == 9 || v8 == 10 && (v4[192] & 1) != 0 )
    v6[192] |= 1u;
  v9 = *(_BYTE *)(*(_QWORD *)(a1 + 176) + 171LL);
  *(v6 - 8) &= ~8u;
  v6[171] = v9 & 8 | v6[171] & 0xF7;
  *(_QWORD *)(a1 + 176) = v6;
  *(_QWORD *)(a1 + 184) = v3 / v5;
  return v3 / v5;
}
