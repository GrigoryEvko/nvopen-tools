// Function: sub_2CF9460
// Address: 0x2cf9460
//
bool __fastcall sub_2CF9460(__int64 a1)
{
  __int64 v1; // rbx
  __int64 v2; // rax
  bool result; // al
  __int64 v4; // rax

  v1 = *(_QWORD *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
  result = *(_BYTE *)v1 == 3
        && !sub_B2FC80(*(_QWORD *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF)))
        && (v2 = *(_QWORD *)(v1 + 8)) != 0
        && *(_DWORD *)(v2 + 8) >> 8 == 4
        && (v4 = *(_QWORD *)(v1 + 24), *(_BYTE *)(v4 + 8) == 16)
        && sub_BCAC40(**(_QWORD **)(v4 + 16), 8);
  return result;
}
