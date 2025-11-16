// Function: sub_1DF1EE0
// Address: 0x1df1ee0
//
__int64 __fastcall sub_1DF1EE0(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  int v3; // r8d
  int v4; // r9d
  __int64 v5; // rax
  int v6; // r8d
  int v7; // r9d
  __int64 v8; // rax
  int v9; // r8d
  int v10; // r9d
  __int64 v11; // rax

  v2 = a2 + 112;
  sub_1636A10(a2, a2);
  v5 = *(unsigned int *)(a2 + 120);
  if ( (unsigned int)v5 >= *(_DWORD *)(a2 + 124) )
  {
    sub_16CD150(v2, (const void *)(a2 + 128), 0, 8, v3, v4);
    v5 = *(unsigned int *)(a2 + 120);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 112) + 8 * v5) = &unk_4FC62EC;
  ++*(_DWORD *)(a2 + 120);
  sub_1636A40(a2, (__int64)&unk_4FC6A0C);
  v8 = *(unsigned int *)(a2 + 120);
  if ( (unsigned int)v8 >= *(_DWORD *)(a2 + 124) )
  {
    sub_16CD150(v2, (const void *)(a2 + 128), 0, 8, v6, v7);
    v8 = *(unsigned int *)(a2 + 120);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 112) + 8 * v8) = &unk_4FC6A0C;
  ++*(_DWORD *)(a2 + 120);
  sub_1636A40(a2, (__int64)&unk_4FC820C);
  v11 = *(unsigned int *)(a2 + 120);
  if ( (unsigned int)v11 >= *(_DWORD *)(a2 + 124) )
  {
    sub_16CD150(v2, (const void *)(a2 + 128), 0, 8, v9, v10);
    v11 = *(unsigned int *)(a2 + 120);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 112) + 8 * v11) = &unk_4FC820C;
  ++*(_DWORD *)(a2 + 120);
  return sub_1E11F70(a1, a2);
}
