// Function: sub_1E1CAF0
// Address: 0x1e1caf0
//
void __fastcall sub_1E1CAF0(__int64 a1, __int64 a2)
{
  int v2; // r8d
  int v3; // r9d
  __int64 v4; // rax
  int v5; // r8d
  int v6; // r9d
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // [rsp+8h] [rbp-38h]

  sub_1636A40(a2, (__int64)&unk_4FC6A0C);
  sub_1636A40(a2, (__int64)&unk_4FC62EC);
  sub_1636A40(a2, (__int64)&unk_4F96DB4);
  sub_1636A40(a2, (__int64)&unk_4FB9E2C);
  v9 = a2 + 112;
  v4 = *(unsigned int *)(a2 + 120);
  if ( (unsigned int)v4 >= *(_DWORD *)(a2 + 124) )
  {
    sub_16CD150(v9, (const void *)(a2 + 128), 0, 8, v2, v3);
    v4 = *(unsigned int *)(a2 + 120);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 112) + 8 * v4) = &unk_4FB9E2C;
  ++*(_DWORD *)(a2 + 120);
  sub_1636A40(a2, (__int64)&unk_4FD4138);
  v7 = *(unsigned int *)(a2 + 120);
  if ( (unsigned int)v7 >= *(_DWORD *)(a2 + 124) )
  {
    sub_16CD150(v9, (const void *)(a2 + 128), 0, 8, v5, v6);
    v7 = *(unsigned int *)(a2 + 120);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 112) + 8 * v7) = &unk_4FC6A0C;
  v8 = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
  *(_DWORD *)(a2 + 120) = v8;
  if ( *(_DWORD *)(a2 + 124) <= (unsigned int)v8 )
  {
    sub_16CD150(v9, (const void *)(a2 + 128), 0, 8, v5, v6);
    v8 = *(unsigned int *)(a2 + 120);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 112) + 8 * v8) = &unk_4FC62EC;
  ++*(_DWORD *)(a2 + 120);
  sub_1E11F70(a1, a2);
}
