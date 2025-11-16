// Function: sub_2098380
// Address: 0x2098380
//
unsigned __int64 __fastcall sub_2098380(__int64 a1)
{
  unsigned __int64 v1; // rdi
  __int64 v2; // rdx
  _QWORD *v3; // rax
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rcx
  _QWORD *v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rcx
  _QWORD *v10; // rax

  v1 = (a1 & 0xFFFFFFFFFFFFFFF8LL) - 24LL * (*(_DWORD *)((a1 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF);
  v2 = *(_QWORD *)(v1 + 72);
  v3 = *(_QWORD **)(v2 + 24);
  if ( *(_DWORD *)(v2 + 32) > 0x40u )
    v3 = (_QWORD *)*v3;
  v4 = 24LL * (int)v3;
  v5 = *(_QWORD *)(v1 + v4 + 120);
  v6 = v4 + 120;
  v7 = *(_QWORD **)(v5 + 24);
  if ( *(_DWORD *)(v5 + 32) > 0x40u )
    v7 = (_QWORD *)*v7;
  v8 = v6 + 8 * (3LL * (int)v7 + 3);
  v9 = *(_QWORD *)(v1 + v8);
  v10 = *(_QWORD **)(v9 + 24);
  if ( *(_DWORD *)(v9 + 32) > 0x40u )
    v10 = (_QWORD *)*v10;
  return v1 + v8 + 8 * (3LL * (int)v10 + 3);
}
