// Function: sub_1DC92A0
// Address: 0x1dc92a0
//
__int64 __fastcall sub_1DC92A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 v6; // rax

  v6 = *(unsigned int *)(a2 + 120);
  *(_BYTE *)(a2 + 160) = 1;
  if ( (unsigned int)v6 >= *(_DWORD *)(a2 + 124) )
  {
    sub_16CD150(a2 + 112, (const void *)(a2 + 128), 0, 8, a5, a6);
    v6 = *(unsigned int *)(a2 + 120);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 112) + 8 * v6) = &unk_4FCA82C;
  ++*(_DWORD *)(a2 + 120);
  sub_1636A90(a2, (__int64)&unk_4FCA82C);
  return sub_1E11F70(a1, a2);
}
