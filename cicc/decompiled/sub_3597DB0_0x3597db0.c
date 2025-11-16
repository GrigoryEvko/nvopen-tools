// Function: sub_3597DB0
// Address: 0x3597db0
//
float __fastcall sub_3597DB0(__int64 a1, __int64 a2)
{
  unsigned int v2; // eax
  unsigned int v3; // edx

  v2 = sub_2E0B010(a2);
  v3 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 920LL) + 8LL * (*(_DWORD *)(a2 + 112) & 0x7FFFFFFF));
  ***(_QWORD ***)(*(_QWORD *)(a1 + 144) + 24LL) = v2;
  **(_QWORD **)(*(_QWORD *)(*(_QWORD *)(a1 + 144) + 24LL) + 8LL) = v3;
  **(_DWORD **)(*(_QWORD *)(*(_QWORD *)(a1 + 144) + 24LL) + 16LL) = *(_DWORD *)(a2 + 116);
  return *(float *)(*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 144) + 24LL))(*(_QWORD *)(a1 + 144));
}
