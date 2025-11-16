// Function: sub_2203160
// Address: 0x2203160
//
void __fastcall sub_2203160(__int64 a1, __int64 a2)
{
  int v2; // r8d
  int v3; // r9d
  __int64 v4; // rax

  sub_1636A10(a2, a2);
  sub_1636A40(a2, (__int64)&unk_4FC6A0C);
  v4 = *(unsigned int *)(a2 + 120);
  if ( (unsigned int)v4 >= *(_DWORD *)(a2 + 124) )
  {
    sub_16CD150(a2 + 112, (const void *)(a2 + 128), 0, 8, v2, v3);
    v4 = *(unsigned int *)(a2 + 120);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 112) + 8 * v4) = &unk_4FC6A0C;
  ++*(_DWORD *)(a2 + 120);
  sub_1E11F70(a1, a2);
}
