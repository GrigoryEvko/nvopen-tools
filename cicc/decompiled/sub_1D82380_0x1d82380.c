// Function: sub_1D82380
// Address: 0x1d82380
//
__int64 __fastcall sub_1D82380(__int64 a1, __int64 a2)
{
  int v2; // r8d
  int v3; // r9d
  __int64 v4; // rax
  int v5; // r8d
  int v6; // r9d
  __int64 v7; // rax
  int v8; // r8d
  int v9; // r9d
  __int64 v10; // rax

  sub_1636A40(a2, (__int64)&unk_4FC5828);
  sub_1636A40(a2, (__int64)&unk_4FC62EC);
  v4 = *(unsigned int *)(a2 + 120);
  if ( (unsigned int)v4 >= *(_DWORD *)(a2 + 124) )
  {
    sub_16CD150(a2 + 112, (const void *)(a2 + 128), 0, 8, v2, v3);
    v4 = *(unsigned int *)(a2 + 120);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 112) + 8 * v4) = &unk_4FC62EC;
  ++*(_DWORD *)(a2 + 120);
  sub_1636A40(a2, (__int64)&unk_4FC6A0C);
  v7 = *(unsigned int *)(a2 + 120);
  if ( (unsigned int)v7 >= *(_DWORD *)(a2 + 124) )
  {
    sub_16CD150(a2 + 112, (const void *)(a2 + 128), 0, 8, v5, v6);
    v7 = *(unsigned int *)(a2 + 120);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 112) + 8 * v7) = &unk_4FC6A0C;
  ++*(_DWORD *)(a2 + 120);
  sub_1636A40(a2, (__int64)&unk_4FC820C);
  v10 = *(unsigned int *)(a2 + 120);
  if ( (unsigned int)v10 >= *(_DWORD *)(a2 + 124) )
  {
    sub_16CD150(a2 + 112, (const void *)(a2 + 128), 0, 8, v8, v9);
    v10 = *(unsigned int *)(a2 + 120);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 112) + 8 * v10) = &unk_4FC820C;
  ++*(_DWORD *)(a2 + 120);
  return sub_1E11F70(a1, a2);
}
