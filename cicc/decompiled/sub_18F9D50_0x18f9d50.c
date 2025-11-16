// Function: sub_18F9D50
// Address: 0x18f9d50
//
void __fastcall sub_18F9D50(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  int v3; // r8d
  int v4; // r9d
  __int64 v5; // rax
  __int64 v6; // rax

  v2 = a2 + 112;
  sub_1636A40(a2, (__int64)&unk_4F9E06C);
  sub_1636A40(a2, (__int64)&unk_4F9D3C0);
  sub_1636A10(a2, (__int64)&unk_4F9D3C0);
  v5 = *(unsigned int *)(a2 + 120);
  if ( (unsigned int)v5 >= *(_DWORD *)(a2 + 124) )
  {
    sub_16CD150(v2, (const void *)(a2 + 128), 0, 8, v3, v4);
    v5 = *(unsigned int *)(a2 + 120);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 112) + 8 * v5) = &unk_4F9E06C;
  v6 = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
  *(_DWORD *)(a2 + 120) = v6;
  if ( *(_DWORD *)(a2 + 124) <= (unsigned int)v6 )
  {
    sub_16CD150(v2, (const void *)(a2 + 128), 0, 8, v3, v4);
    v6 = *(unsigned int *)(a2 + 120);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 112) + 8 * v6) = &unk_4F98E5C;
  ++*(_DWORD *)(a2 + 120);
  nullsub_571();
}
