// Function: sub_E290B0
// Address: 0xe290b0
//
unsigned __int64 __fastcall sub_E290B0(__int64 a1, __int64 *a2, char a3)
{
  _QWORD *v4; // rax
  unsigned __int64 v5; // r12
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // r15
  __int64 *v9; // rax
  __int64 *v10; // r15
  __int64 v11; // rax
  __int64 v12; // rax
  char v13; // bl
  char v14; // bl

  v4 = *(_QWORD **)(a1 + 16);
  v5 = (*v4 + v4[1] + 7LL) & 0xFFFFFFFFFFFFFFF8LL;
  v4[1] = v5 - *v4 + 40;
  if ( *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) > *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL) )
  {
    v9 = (__int64 *)sub_22077B0(32);
    v10 = v9;
    if ( v9 )
    {
      *v9 = 0;
      v9[1] = 0;
      v9[2] = 0;
      v9[3] = 0;
    }
    v11 = sub_2207820(4096);
    v10[2] = 4096;
    *v10 = v11;
    v5 = v11;
    v12 = *(_QWORD *)(a1 + 16);
    v10[1] = 40;
    v10[3] = v12;
    *(_QWORD *)(a1 + 16) = v10;
  }
  if ( !v5 )
  {
    MEMORY[0x20] = sub_E27700(a1, a2, 0);
    BUG();
  }
  *(_DWORD *)(v5 + 8) = 27;
  *(_BYTE *)(v5 + 24) = 0;
  *(_QWORD *)(v5 + 16) = 0;
  *(_QWORD *)v5 = &unk_49E11E0;
  *(_QWORD *)(v5 + 32) = 0;
  v6 = sub_E27700(a1, a2, 0);
  *(_BYTE *)(v5 + 24) = a3;
  *(_QWORD *)(v5 + 32) = v6;
  v7 = v6;
  if ( *(_BYTE *)(a1 + 8) )
    return 0;
  if ( *(_DWORD *)(v6 + 8) == 14 )
  {
    v13 = *(_BYTE *)(v6 + 12);
    *(_BYTE *)(v6 + 12) = sub_E24310(a1, a2) | v13;
    v14 = sub_E22E40(a1, a2);
    if ( *(_QWORD *)(v7 + 24) )
      sub_E27270(a1, a2);
    *(_BYTE *)(*(_QWORD *)(v7 + 32) + 12LL) |= v14;
  }
  else
  {
    *(_BYTE *)(*(_QWORD *)(v5 + 32) + 12LL) = sub_E22E40(a1, a2);
  }
  return v5;
}
