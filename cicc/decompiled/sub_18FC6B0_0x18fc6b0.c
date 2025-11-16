// Function: sub_18FC6B0
// Address: 0x18fc6b0
//
__int64 __fastcall sub_18FC6B0(__int64 a1, __int64 a2)
{
  __int64 v3; // r13
  void *v4; // rsi
  int v5; // r8d
  int v6; // r9d
  __int64 v7; // rax
  __int64 v8; // rax

  v3 = a2 + 112;
  sub_1636A40(a2, (__int64)&unk_4F9D764);
  sub_1636A40(a2, (__int64)&unk_4F9E06C);
  sub_1636A40(a2, (__int64)&unk_4F9B6E8);
  sub_1636A40(a2, (__int64)&unk_4F9D3C0);
  v4 = &unk_4F99768;
  sub_1636A40(a2, (__int64)&unk_4F99768);
  v7 = *(unsigned int *)(a2 + 120);
  if ( (unsigned int)v7 >= *(_DWORD *)(a2 + 124) )
  {
    v4 = (void *)(a2 + 128);
    sub_16CD150(v3, (const void *)(a2 + 128), 0, 8, v5, v6);
    v7 = *(unsigned int *)(a2 + 120);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 112) + 8 * v7) = &unk_4F99768;
  v8 = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
  *(_DWORD *)(a2 + 120) = v8;
  if ( *(_DWORD *)(a2 + 124) <= (unsigned int)v8 )
  {
    v4 = (void *)(a2 + 128);
    sub_16CD150(v3, (const void *)(a2 + 128), 0, 8, v5, v6);
    v8 = *(unsigned int *)(a2 + 120);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 112) + 8 * v8) = &unk_4F98E5C;
  ++*(_DWORD *)(a2 + 120);
  return sub_1636A10(a2, (__int64)v4);
}
