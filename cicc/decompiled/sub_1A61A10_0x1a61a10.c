// Function: sub_1A61A10
// Address: 0x1a61a10
//
__int64 __fastcall sub_1A61A10(__int64 a1, __int64 a2)
{
  int v2; // r8d
  int v3; // r9d
  __int64 v4; // rax

  sub_1636A40(a2, (__int64)&unk_4F9D764);
  sub_1636A40(a2, (__int64)&unk_4F9D3C0);
  v4 = *(unsigned int *)(a2 + 120);
  if ( (unsigned int)v4 >= *(_DWORD *)(a2 + 124) )
  {
    sub_16CD150(a2 + 112, (const void *)(a2 + 128), 0, 8, v2, v3);
    v4 = *(unsigned int *)(a2 + 120);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 112) + 8 * v4) = &unk_4F98E5C;
  ++*(_DWORD *)(a2 + 120);
  sub_1636A40(a2, (__int64)&unk_4FBA0D1);
  return sub_1636A40(a2, (__int64)&unk_4FB9E3C);
}
