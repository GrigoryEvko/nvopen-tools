// Function: sub_1F56670
// Address: 0x1f56670
//
void __fastcall sub_1F56670(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 v6; // r14
  __int64 v7; // rax
  __int64 v8; // rax

  v6 = a2 + 112;
  v7 = *(unsigned int *)(a2 + 120);
  if ( (unsigned int)v7 >= *(_DWORD *)(a2 + 124) )
  {
    sub_16CD150(v6, (const void *)(a2 + 128), 0, 8, a5, a6);
    v7 = *(unsigned int *)(a2 + 120);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 112) + 8 * v7) = &unk_4FC6A0C;
  v8 = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
  *(_DWORD *)(a2 + 120) = v8;
  if ( *(_DWORD *)(a2 + 124) <= (unsigned int)v8 )
  {
    sub_16CD150(v6, (const void *)(a2 + 128), 0, 8, a5, a6);
    v8 = *(unsigned int *)(a2 + 120);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 112) + 8 * v8) = &unk_4FC62EC;
  ++*(_DWORD *)(a2 + 120);
  sub_1E11F70(a1, a2);
}
