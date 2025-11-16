// Function: sub_E28F00
// Address: 0xe28f00
//
unsigned __int64 __fastcall sub_E28F00(__int64 a1, size_t *a2)
{
  _QWORD *v2; // rax
  unsigned __int64 v3; // r12
  __int64 v4; // rax
  char v6; // bl
  unsigned __int64 v7; // rax
  __int64 *v8; // rax
  __int64 *v9; // rbx
  __int64 v10; // rax
  __int64 v11; // rax

  v2 = *(_QWORD **)(a1 + 16);
  v3 = (*v2 + v2[1] + 7LL) & 0xFFFFFFFFFFFFFFF8LL;
  v2[1] = v3 - *v2 + 40;
  if ( *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) > *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL) )
  {
    v8 = (__int64 *)sub_22077B0(32);
    v9 = v8;
    if ( v8 )
    {
      *v8 = 0;
      v8[1] = 0;
      v8[2] = 0;
      v8[3] = 0;
    }
    v10 = sub_2207820(4096);
    v9[2] = 4096;
    *v9 = v10;
    v3 = v10;
    v11 = *(_QWORD *)(a1 + 16);
    v9[1] = 40;
    v9[3] = v11;
    *(_QWORD *)(a1 + 16) = v9;
  }
  if ( !v3 )
  {
    sub_E207A0(a2);
    MEMORY[0xC] = 0;
    BUG();
  }
  *(_BYTE *)(v3 + 12) = 0;
  *(_DWORD *)(v3 + 8) = 14;
  *(_DWORD *)(v3 + 16) = 0;
  *(_QWORD *)v3 = &unk_49E10E8;
  *(_QWORD *)(v3 + 24) = 0;
  *(_QWORD *)(v3 + 32) = 0;
  v4 = sub_E207A0(a2);
  *(_BYTE *)(v3 + 12) = v4;
  *(_DWORD *)(v3 + 16) = HIDWORD(v4);
  *(_BYTE *)(v3 + 12) |= sub_E24310(a1, (__int64 *)a2);
  if ( (unsigned __int8)sub_E20730(a2, 1u, "8") )
  {
    *(_QWORD *)(v3 + 24) = sub_E27270(a1, (__int64 *)a2);
    *(_QWORD *)(v3 + 32) = sub_E28570(a1, (__int64 *)a2, 1);
    return v3;
  }
  v6 = sub_E22E40(a1, (__int64 *)a2);
  *(_QWORD *)(v3 + 24) = sub_E27270(a1, (__int64 *)a2);
  v7 = sub_E27700(a1, (__int64 *)a2, 0);
  *(_QWORD *)(v3 + 32) = v7;
  if ( !v7 )
    return v3;
  *(_BYTE *)(v7 + 12) = v6;
  return v3;
}
