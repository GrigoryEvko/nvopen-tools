// Function: sub_1D46220
// Address: 0x1d46220
//
__int64 __fastcall sub_1D46220(__int64 a1, __int64 a2)
{
  int v2; // r8d
  int v3; // r9d
  __int64 v4; // rax

  if ( *(_DWORD *)(a1 + 304) )
    sub_1636A40(a2, (__int64)&unk_4F96DB4);
  sub_1636A40(a2, (__int64)&unk_4FC3606);
  v4 = *(unsigned int *)(a2 + 120);
  if ( (unsigned int)v4 >= *(_DWORD *)(a2 + 124) )
  {
    sub_16CD150(a2 + 112, (const void *)(a2 + 128), 0, 8, v2, v3);
    v4 = *(unsigned int *)(a2 + 120);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 112) + 8 * v4) = &unk_4FC3606;
  ++*(_DWORD *)(a2 + 120);
  sub_1636A40(a2, (__int64)&unk_4F9B6E8);
  sub_1636A40(a2, (__int64)&unk_4F9D3C0);
  if ( byte_4FC1CC0 && *(_DWORD *)(a1 + 304) )
    sub_1636A40(a2, (__int64)&unk_4F98724);
  return sub_1E11F70(a1, a2);
}
