// Function: sub_1A0F350
// Address: 0x1a0f350
//
__int64 __fastcall sub_1A0F350(__int64 a1, __int64 a2)
{
  void *v4; // rsi
  int v5; // r8d
  int v6; // r9d
  __int64 v7; // rax

  v4 = &unk_4F9B6E8;
  sub_1636A40(a2, (__int64)&unk_4F9B6E8);
  v7 = *(unsigned int *)(a2 + 120);
  if ( (unsigned int)v7 >= *(_DWORD *)(a2 + 124) )
  {
    v4 = (void *)(a2 + 128);
    sub_16CD150(a2 + 112, (const void *)(a2 + 128), 0, 8, v5, v6);
    v7 = *(unsigned int *)(a2 + 120);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 112) + 8 * v7) = &unk_4F98E5C;
  ++*(_DWORD *)(a2 + 120);
  return sub_1636A10(a2, (__int64)v4);
}
