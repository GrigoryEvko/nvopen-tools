// Function: sub_210DE30
// Address: 0x210de30
//
void __fastcall sub_210DE30(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 v6; // rax

  v6 = *(unsigned int *)(a2 + 120);
  if ( (unsigned int)v6 >= *(_DWORD *)(a2 + 124) )
  {
    sub_16CD150(a2 + 112, (const void *)(a2 + 128), 0, 8, a5, a6);
    v6 = *(unsigned int *)(a2 + 120);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 112) + 8 * v6) = &unk_4FCABAC;
  ++*(_DWORD *)(a2 + 120);
  sub_1E11F70(a1, a2);
}
