// Function: sub_1AE8290
// Address: 0x1ae8290
//
bool __fastcall sub_1AE8290(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v3; // rax
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // r12
  __int64 v8; // rax
  unsigned __int64 v9; // r15
  __int64 v10; // rax
  _QWORD *v11; // rbx
  __int64 v12; // rdx

  v3 = (_QWORD *)(a3 + 24);
  if ( !a3 )
    v3 = 0;
  if ( *(_QWORD **)(*(_QWORD *)(a3 + 40) + 48LL) == v3 )
    return 0;
  v5 = *v3 & 0xFFFFFFFFFFFFFFF8LL;
  v6 = v5;
  if ( !v5 )
    BUG();
  if ( *(_BYTE *)(v5 - 8) == 78
    && (v8 = *(_QWORD *)(v5 - 48), !*(_BYTE *)(v8 + 16))
    && (*(_BYTE *)(v8 + 33) & 0x20) != 0
    && *(_DWORD *)(v8 + 36) == 38
    && ((v9 = v6 - 24, v10 = sub_1601A30(v6 - 24, 0), (*(_BYTE *)(a3 + 23) & 0x40) == 0)
      ? (v11 = (_QWORD *)(a3 - 24LL * (*(_DWORD *)(a3 + 20) & 0xFFFFFFF)))
      : (v11 = *(_QWORD **)(a3 - 8)),
        *v11 == v10
     && (v12 = *(_DWORD *)(v6 - 4) & 0xFFFFFFF, a1 == *(_QWORD *)(*(_QWORD *)(v9 + 24 * (1 - v12)) + 24LL))) )
  {
    return *(_QWORD *)(*(_QWORD *)(v9 + 24 * (2 - v12)) + 24LL) == a2;
  }
  else
  {
    return 0;
  }
}
