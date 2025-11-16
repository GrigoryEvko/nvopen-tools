// Function: sub_1A95F60
// Address: 0x1a95f60
//
unsigned __int64 __fastcall sub_1A95F60(__int64 a1)
{
  unsigned __int64 v1; // rdi
  unsigned __int64 v2; // rdx
  __int64 v3; // rcx
  _QWORD *v4; // rax
  __int64 v5; // rax
  __int64 v6; // rcx
  __int64 v7; // rsi
  _QWORD *v8; // rax
  __int64 v9; // rcx
  __int64 v10; // rsi
  _QWORD *v11; // rax
  unsigned __int64 v12; // rax

  v1 = a1 & 0xFFFFFFFFFFFFFFF8LL;
  v2 = v1 - 24LL * (*(_DWORD *)(v1 + 20) & 0xFFFFFFF);
  v3 = *(_QWORD *)(v2 + 72);
  v4 = *(_QWORD **)(v3 + 24);
  if ( *(_DWORD *)(v3 + 32) > 0x40u )
    v4 = (_QWORD *)*v4;
  v5 = 24LL * (int)v4;
  v6 = *(_QWORD *)(v2 + v5 + 120);
  v7 = v5 + 120;
  v8 = *(_QWORD **)(v6 + 24);
  if ( *(_DWORD *)(v6 + 32) > 0x40u )
    v8 = (_QWORD *)*v8;
  v9 = v7 + 8 * (3LL * (int)v8 + 3);
  v10 = *(_QWORD *)(v2 + v9);
  v11 = *(_QWORD **)(v10 + 24);
  if ( *(_DWORD *)(v10 + 32) > 0x40u )
    v11 = (_QWORD *)*v11;
  v12 = v2 + v9 + 8 * (3LL * (int)v11 + 3);
  if ( (*(_BYTE *)(v1 + 23) & 0x40) != 0 )
    v2 = *(_QWORD *)(v1 - 8);
  return 0xAAAAAAAAAAAAAAABLL * ((__int64)(v12 - v2) >> 3);
}
