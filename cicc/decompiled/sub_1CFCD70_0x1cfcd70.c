// Function: sub_1CFCD70
// Address: 0x1cfcd70
//
void __fastcall sub_1CFCD70(__int64 a1, __int64 a2, int a3, __int64 a4, __int64 a5, int a6)
{
  __int64 v6; // rbx
  __int64 i; // r14
  unsigned __int64 v9; // r12
  bool v10; // zf
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rdx
  int v14; // [rsp+Ch] [rbp-34h]

  v6 = *(_QWORD *)(a2 + 32);
  for ( i = v6 + 16LL * *(unsigned int *)(a2 + 40); i != v6; v6 += 16 )
  {
    v9 = *(_QWORD *)v6 & 0xFFFFFFFFFFFFFFF8LL;
    v10 = (*(_DWORD *)(v9 + 212))-- == 1;
    if ( v10 && v9 != a1 + 72 )
    {
      *(_BYTE *)(v9 + 229) |= 2u;
      v11 = *(unsigned int *)(a1 + 672);
      if ( (unsigned int)v11 >= *(_DWORD *)(a1 + 676) )
      {
        v14 = a3;
        sub_16CD150(a1 + 664, (const void *)(a1 + 680), 0, 8, a3, a6);
        v11 = *(unsigned int *)(a1 + 672);
        a3 = v14;
      }
      *(_QWORD *)(*(_QWORD *)(a1 + 664) + 8 * v11) = v9;
      ++*(_DWORD *)(a1 + 672);
    }
    if ( (*(_BYTE *)v6 & 6) == 0 )
    {
      v12 = *(unsigned int *)(v6 + 8);
      if ( (_DWORD)v12 )
      {
        v13 = *(_QWORD *)(a1 + 816);
        if ( !*(_QWORD *)(v13 + 8 * v12) )
        {
          ++*(_DWORD *)(a1 + 808);
          *(_QWORD *)(v13 + 8LL * *(unsigned int *)(v6 + 8)) = *(_QWORD *)v6 & 0xFFFFFFFFFFFFFFF8LL;
          *(_DWORD *)(*(_QWORD *)(a1 + 840) + 4LL * *(unsigned int *)(v6 + 8)) = a3;
        }
      }
    }
  }
}
