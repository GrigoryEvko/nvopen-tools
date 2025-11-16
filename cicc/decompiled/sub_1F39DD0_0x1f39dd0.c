// Function: sub_1F39DD0
// Address: 0x1f39dd0
//
bool __fastcall sub_1F39DD0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 v4; // r13
  __int64 v5; // r12
  __int64 v7; // r14
  int v8; // esi
  int v9; // esi
  bool result; // al
  __int64 v11; // rax

  v3 = *(_QWORD *)(a3 + 56);
  v4 = 0;
  v5 = *(_QWORD *)(a2 + 32);
  v7 = *(_QWORD *)(v3 + 40);
  if ( !*(_BYTE *)(v5 + 40) )
  {
    v8 = *(_DWORD *)(v5 + 48);
    if ( v8 < 0 )
      v4 = sub_1E69D60(*(_QWORD *)(v3 + 40), v8);
  }
  result = !*(_BYTE *)(v5 + 80)
        && (v9 = *(_DWORD *)(v5 + 88), v9 < 0)
        && (v11 = sub_1E69D60(v7, v9), v4)
        && v11
        && a3 == *(_QWORD *)(v4 + 24)
        && *(_QWORD *)(v11 + 24) == a3;
  return result;
}
