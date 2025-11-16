// Function: sub_1DB9C00
// Address: 0x1db9c00
//
__int64 __fastcall sub_1DB9C00(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  int v3; // r8d
  int v4; // r9d
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rax
  int v8; // r8d
  int v9; // r9d
  __int64 v10; // rax
  __int64 v11; // rax

  v2 = a2 + 112;
  sub_1636A10(a2, a2);
  sub_1636A40(a2, (__int64)&unk_4F96DB4);
  v5 = *(unsigned int *)(a2 + 120);
  if ( (unsigned int)v5 >= *(_DWORD *)(a2 + 124) )
  {
    sub_16CD150(v2, (const void *)(a2 + 128), 0, 8, v3, v4);
    v5 = *(unsigned int *)(a2 + 120);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 112) + 8 * v5) = &unk_4F96DB4;
  v6 = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
  *(_DWORD *)(a2 + 120) = v6;
  if ( *(_DWORD *)(a2 + 124) <= (unsigned int)v6 )
  {
    sub_16CD150(v2, (const void *)(a2 + 128), 0, 8, v3, v4);
    v6 = *(unsigned int *)(a2 + 120);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 112) + 8 * v6) = &unk_4FC4534;
  v7 = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
  *(_DWORD *)(a2 + 120) = v7;
  if ( *(_DWORD *)(a2 + 124) <= (unsigned int)v7 )
  {
    sub_16CD150(v2, (const void *)(a2 + 128), 0, 8, v3, v4);
    v7 = *(unsigned int *)(a2 + 120);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 112) + 8 * v7) = &unk_4FC6A0C;
  ++*(_DWORD *)(a2 + 120);
  sub_1636A90(a2, (__int64)&unk_4FC62EC);
  v10 = *(unsigned int *)(a2 + 120);
  if ( (unsigned int)v10 >= *(_DWORD *)(a2 + 124) )
  {
    sub_16CD150(v2, (const void *)(a2 + 128), 0, 8, v8, v9);
    v10 = *(unsigned int *)(a2 + 120);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 112) + 8 * v10) = &unk_4FC62EC;
  v11 = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
  *(_DWORD *)(a2 + 120) = v11;
  if ( *(_DWORD *)(a2 + 124) <= (unsigned int)v11 )
  {
    sub_16CD150(v2, (const void *)(a2 + 128), 0, 8, v8, v9);
    v11 = *(unsigned int *)(a2 + 120);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 112) + 8 * v11) = &unk_4FCA82C;
  ++*(_DWORD *)(a2 + 120);
  sub_1636A90(a2, (__int64)&unk_4FCA82C);
  return sub_1E11F70(a1, a2);
}
