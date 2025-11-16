// Function: sub_E287D0
// Address: 0xe287d0
//
unsigned __int64 __fastcall sub_E287D0(__int64 a1, size_t *a2)
{
  _QWORD *v2; // rax
  unsigned __int64 v3; // r12
  __int64 v4; // rax
  __int64 *v6; // rax
  __int64 *v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rax

  v2 = *(_QWORD **)(a1 + 16);
  v3 = (*v2 + v2[1] + 7LL) & 0xFFFFFFFFFFFFFFF8LL;
  v2[1] = v3 - *v2 + 40;
  if ( *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) > *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL) )
  {
    v6 = (__int64 *)sub_22077B0(32);
    v7 = v6;
    if ( v6 )
    {
      *v6 = 0;
      v6[1] = 0;
      v6[2] = 0;
      v6[3] = 0;
    }
    v8 = sub_2207820(4096);
    v7[2] = 4096;
    *v7 = v8;
    v3 = v8;
    v9 = *(_QWORD *)(a1 + 16);
    v7[1] = 40;
    v7[3] = v9;
    *(_QWORD *)(a1 + 16) = v7;
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
  if ( (unsigned __int8)sub_E20730(a2, 1u, "6") )
  {
    *(_QWORD *)(v3 + 32) = sub_E28570(a1, (__int64 *)a2, 0);
  }
  else
  {
    *(_BYTE *)(v3 + 12) |= sub_E24310(a1, (__int64 *)a2);
    *(_QWORD *)(v3 + 32) = sub_E27700(a1, (__int64 *)a2, 1);
  }
  return v3;
}
