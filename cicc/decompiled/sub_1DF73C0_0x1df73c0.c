// Function: sub_1DF73C0
// Address: 0x1df73c0
//
__int64 __fastcall sub_1DF73C0(__int64 a1, __int64 a2)
{
  int v2; // r8d
  int v3; // r9d
  __int64 v4; // rax
  int v5; // r8d
  int v6; // r9d
  __int64 v7; // rax
  int v8; // r8d
  int v9; // r9d
  __int64 result; // rax

  sub_1636A10(a2, a2);
  sub_1E11F70(a1, a2);
  sub_1636A40(a2, (__int64)&unk_4F96DB4);
  v4 = *(unsigned int *)(a2 + 120);
  if ( (unsigned int)v4 >= *(_DWORD *)(a2 + 124) )
  {
    sub_16CD150(a2 + 112, (const void *)(a2 + 128), 0, 8, v2, v3);
    v4 = *(unsigned int *)(a2 + 120);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 112) + 8 * v4) = &unk_4FC6A0C;
  ++*(_DWORD *)(a2 + 120);
  sub_1636A40(a2, (__int64)&unk_4FC62EC);
  v7 = *(unsigned int *)(a2 + 120);
  if ( (unsigned int)v7 >= *(_DWORD *)(a2 + 124) )
  {
    sub_16CD150(a2 + 112, (const void *)(a2 + 128), 0, 8, v5, v6);
    v7 = *(unsigned int *)(a2 + 120);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 112) + 8 * v7) = &unk_4FC62EC;
  ++*(_DWORD *)(a2 + 120);
  sub_1636A40(a2, (__int64)&unk_4FD4138);
  sub_1636A40(a2, (__int64)&unk_4FB9E2C);
  result = *(unsigned int *)(a2 + 120);
  if ( (unsigned int)result >= *(_DWORD *)(a2 + 124) )
  {
    sub_16CD150(a2 + 112, (const void *)(a2 + 128), 0, 8, v8, v9);
    result = *(unsigned int *)(a2 + 120);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 112) + 8 * result) = &unk_4FB9E2C;
  ++*(_DWORD *)(a2 + 120);
  return result;
}
