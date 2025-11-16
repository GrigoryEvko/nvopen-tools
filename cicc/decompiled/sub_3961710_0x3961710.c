// Function: sub_3961710
// Address: 0x3961710
//
__int64 __fastcall sub_3961710(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  int v3; // r8d
  int v4; // r9d
  __int64 v5; // rax
  int v6; // r8d
  int v7; // r9d
  __int64 v8; // rax
  int v9; // r8d
  int v10; // r9d
  __int64 v11; // rax
  int v12; // r8d
  int v13; // r9d
  __int64 v14; // rax
  int v15; // r8d
  int v16; // r9d
  __int64 result; // rax

  v2 = a2 + 112;
  sub_1636A10(a2, a2);
  sub_1636A40(a2, (__int64)&unk_4F9920C);
  v5 = *(unsigned int *)(a2 + 120);
  if ( (unsigned int)v5 >= *(_DWORD *)(a2 + 124) )
  {
    sub_16CD150(v2, (const void *)(a2 + 128), 0, 8, v3, v4);
    v5 = *(unsigned int *)(a2 + 120);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 112) + 8 * v5) = &unk_4F9920C;
  ++*(_DWORD *)(a2 + 120);
  sub_1636A40(a2, (__int64)&unk_4F9E06C);
  v8 = *(unsigned int *)(a2 + 120);
  if ( (unsigned int)v8 >= *(_DWORD *)(a2 + 124) )
  {
    sub_16CD150(v2, (const void *)(a2 + 128), 0, 8, v6, v7);
    v8 = *(unsigned int *)(a2 + 120);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 112) + 8 * v8) = &unk_4F9E06C;
  ++*(_DWORD *)(a2 + 120);
  sub_1636A40(a2, (__int64)&unk_4F97E48);
  v11 = *(unsigned int *)(a2 + 120);
  if ( (unsigned int)v11 >= *(_DWORD *)(a2 + 124) )
  {
    sub_16CD150(v2, (const void *)(a2 + 128), 0, 8, v9, v10);
    v11 = *(unsigned int *)(a2 + 120);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 112) + 8 * v11) = &unk_4F97E48;
  ++*(_DWORD *)(a2 + 120);
  sub_1636A40(a2, (__int64)&unk_4F9D3C0);
  v14 = *(unsigned int *)(a2 + 120);
  if ( (unsigned int)v14 >= *(_DWORD *)(a2 + 124) )
  {
    sub_16CD150(v2, (const void *)(a2 + 128), 0, 8, v12, v13);
    v14 = *(unsigned int *)(a2 + 120);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 112) + 8 * v14) = &unk_4F9D3C0;
  ++*(_DWORD *)(a2 + 120);
  sub_1636A40(a2, (__int64)&unk_4FB9E2C);
  result = *(unsigned int *)(a2 + 120);
  if ( (unsigned int)result >= *(_DWORD *)(a2 + 124) )
  {
    sub_16CD150(v2, (const void *)(a2 + 128), 0, 8, v15, v16);
    result = *(unsigned int *)(a2 + 120);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 112) + 8 * result) = &unk_4FB9E2C;
  ++*(_DWORD *)(a2 + 120);
  return result;
}
