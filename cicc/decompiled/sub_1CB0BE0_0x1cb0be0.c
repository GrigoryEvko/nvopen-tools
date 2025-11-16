// Function: sub_1CB0BE0
// Address: 0x1cb0be0
//
bool __fastcall sub_1CB0BE0(__int64 a1)
{
  __int64 v1; // rbx
  bool result; // al
  __int64 v3; // rax

  v1 = *(_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
  result = *(_BYTE *)(v1 + 16) == 3
        && !sub_15E4F60(*(_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF)))
        && *(_QWORD *)v1
        && *(_DWORD *)(*(_QWORD *)v1 + 8LL) >> 8 == 4
        && (v3 = *(_QWORD *)(v1 + 24), *(_BYTE *)(v3 + 8) == 14)
        && sub_1642F90(**(_QWORD **)(v3 + 16), 8);
  return result;
}
