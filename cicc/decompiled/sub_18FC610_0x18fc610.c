// Function: sub_18FC610
// Address: 0x18fc610
//
__int64 __fastcall sub_18FC610(__int64 a1, __int64 a2)
{
  void *v3; // rsi
  int v4; // r8d
  int v5; // r9d
  __int64 v6; // rax

  sub_1636A40(a2, (__int64)&unk_4F9D764);
  sub_1636A40(a2, (__int64)&unk_4F9E06C);
  sub_1636A40(a2, (__int64)&unk_4F9B6E8);
  v3 = &unk_4F9D3C0;
  sub_1636A40(a2, (__int64)&unk_4F9D3C0);
  v6 = *(unsigned int *)(a2 + 120);
  if ( (unsigned int)v6 >= *(_DWORD *)(a2 + 124) )
  {
    v3 = (void *)(a2 + 128);
    sub_16CD150(a2 + 112, (const void *)(a2 + 128), 0, 8, v4, v5);
    v6 = *(unsigned int *)(a2 + 120);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 112) + 8 * v6) = &unk_4F98E5C;
  ++*(_DWORD *)(a2 + 120);
  return sub_1636A10(a2, (__int64)v3);
}
