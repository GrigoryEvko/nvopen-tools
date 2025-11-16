// Function: sub_1F2F3D0
// Address: 0x1f2f3d0
//
void __fastcall sub_1F2F3D0(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  int v3; // r8d
  int v4; // r9d
  __int64 v5; // rax
  int v6; // r8d
  int v7; // r9d
  __int64 v8; // rax
  __int64 v9; // rax

  v2 = a2 + 112;
  sub_1636A10(a2, a2);
  sub_1636A40(a2, (__int64)&unk_4FCA82C);
  v5 = *(unsigned int *)(a2 + 120);
  if ( (unsigned int)v5 >= *(_DWORD *)(a2 + 124) )
  {
    sub_16CD150(v2, (const void *)(a2 + 128), 0, 8, v3, v4);
    v5 = *(unsigned int *)(a2 + 120);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 112) + 8 * v5) = &unk_4FCA82C;
  ++*(_DWORD *)(a2 + 120);
  sub_1636A40(a2, (__int64)&unk_4FC452C);
  sub_1636A40(a2, (__int64)&unk_4FC453D);
  v8 = *(unsigned int *)(a2 + 120);
  if ( (unsigned int)v8 >= *(_DWORD *)(a2 + 124) )
  {
    sub_16CD150(v2, (const void *)(a2 + 128), 0, 8, v6, v7);
    v8 = *(unsigned int *)(a2 + 120);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 112) + 8 * v8) = &unk_4FC453D;
  v9 = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
  *(_DWORD *)(a2 + 120) = v9;
  if ( *(_DWORD *)(a2 + 124) <= (unsigned int)v9 )
  {
    sub_16CD150(v2, (const void *)(a2 + 128), 0, 8, v6, v7);
    v9 = *(unsigned int *)(a2 + 120);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 112) + 8 * v9) = &unk_4FC62EC;
  ++*(_DWORD *)(a2 + 120);
  sub_1E11F70(a1, a2);
}
