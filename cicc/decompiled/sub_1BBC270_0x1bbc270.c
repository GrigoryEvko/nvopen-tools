// Function: sub_1BBC270
// Address: 0x1bbc270
//
void __fastcall sub_1BBC270(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 v6; // r12
  __int64 v7; // rax

  if ( a2 == *(_QWORD *)(a2 + 8) && !*(_DWORD *)(a2 + 96) && !*(_BYTE *)(a2 + 100) )
  {
    v6 = *a1;
    v7 = *(unsigned int *)(*a1 + 8);
    if ( (unsigned int)v7 >= *(_DWORD *)(*a1 + 12) )
    {
      sub_16CD150(*a1, (const void *)(v6 + 16), 0, 8, a5, a6);
      v7 = *(unsigned int *)(v6 + 8);
    }
    *(_QWORD *)(*(_QWORD *)v6 + 8 * v7) = a2;
    ++*(_DWORD *)(v6 + 8);
  }
}
