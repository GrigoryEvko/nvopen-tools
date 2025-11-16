// Function: sub_1994860
// Address: 0x1994860
//
__int64 __fastcall sub_1994860(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 v6; // r14
  __int64 v7; // rax
  int v8; // r8d
  int v9; // r9d
  __int64 v10; // rax
  int v11; // r8d
  int v12; // r9d
  __int64 v13; // rax
  int v14; // r8d
  int v15; // r9d
  __int64 v16; // rax
  int v17; // r8d
  int v18; // r9d
  __int64 v19; // rax

  v6 = a2 + 112;
  v7 = *(unsigned int *)(a2 + 120);
  if ( (unsigned int)v7 >= *(_DWORD *)(a2 + 124) )
  {
    sub_16CD150(v6, (const void *)(a2 + 128), 0, 8, a5, a6);
    v7 = *(unsigned int *)(a2 + 120);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 112) + 8 * v7) = &unk_4FB66D8;
  ++*(_DWORD *)(a2 + 120);
  sub_1636A40(a2, (__int64)&unk_4F9920C);
  v10 = *(unsigned int *)(a2 + 120);
  if ( (unsigned int)v10 >= *(_DWORD *)(a2 + 124) )
  {
    sub_16CD150(v6, (const void *)(a2 + 128), 0, 8, v8, v9);
    v10 = *(unsigned int *)(a2 + 120);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 112) + 8 * v10) = &unk_4F9920C;
  ++*(_DWORD *)(a2 + 120);
  sub_1636A40(a2, (__int64)&unk_4FB66D8);
  sub_1636A40(a2, (__int64)&unk_4F9E06C);
  v13 = *(unsigned int *)(a2 + 120);
  if ( (unsigned int)v13 >= *(_DWORD *)(a2 + 124) )
  {
    sub_16CD150(v6, (const void *)(a2 + 128), 0, 8, v11, v12);
    v13 = *(unsigned int *)(a2 + 120);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 112) + 8 * v13) = &unk_4F9E06C;
  ++*(_DWORD *)(a2 + 120);
  sub_1636A40(a2, (__int64)&unk_4F9A488);
  v16 = *(unsigned int *)(a2 + 120);
  if ( (unsigned int)v16 >= *(_DWORD *)(a2 + 124) )
  {
    sub_16CD150(v6, (const void *)(a2 + 128), 0, 8, v14, v15);
    v16 = *(unsigned int *)(a2 + 120);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 112) + 8 * v16) = &unk_4F9A488;
  ++*(_DWORD *)(a2 + 120);
  sub_1636A40(a2, (__int64)&unk_4FB66D8);
  sub_1636A40(a2, (__int64)&unk_4F98F4C);
  v19 = *(unsigned int *)(a2 + 120);
  if ( (unsigned int)v19 >= *(_DWORD *)(a2 + 124) )
  {
    sub_16CD150(v6, (const void *)(a2 + 128), 0, 8, v17, v18);
    v19 = *(unsigned int *)(a2 + 120);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 112) + 8 * v19) = &unk_4F98F4C;
  ++*(_DWORD *)(a2 + 120);
  sub_1636A40(a2, (__int64)&unk_4F9D3C0);
  return sub_1636A40(a2, (__int64)&unk_4FB9E2C);
}
