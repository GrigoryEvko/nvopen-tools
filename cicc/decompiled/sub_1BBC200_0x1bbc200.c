// Function: sub_1BBC200
// Address: 0x1bbc200
//
void __fastcall sub_1BBC200(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 v6; // rdx
  __int64 v8; // rbx
  __int64 v9; // r12
  __int64 v10; // rax

  if ( a2 )
  {
    if ( *(_DWORD *)(a2 + 88) != -1 )
    {
      v6 = *(_QWORD *)(a2 + 8);
      --*(_DWORD *)(a2 + 92);
      if ( (*(_DWORD *)(v6 + 96))-- == 1 )
      {
        v8 = *a1;
        v9 = *(_QWORD *)(a2 + 8);
        v10 = *(unsigned int *)(*a1 + 8);
        if ( (unsigned int)v10 >= *(_DWORD *)(*a1 + 12) )
        {
          sub_16CD150(*a1, (const void *)(v8 + 16), 0, 8, a5, a6);
          v10 = *(unsigned int *)(v8 + 8);
        }
        *(_QWORD *)(*(_QWORD *)v8 + 8 * v10) = v9;
        ++*(_DWORD *)(v8 + 8);
      }
    }
  }
}
