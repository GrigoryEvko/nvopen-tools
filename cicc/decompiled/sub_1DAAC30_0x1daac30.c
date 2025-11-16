// Function: sub_1DAAC30
// Address: 0x1daac30
//
void __fastcall sub_1DAAC30(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  __int64 v7; // rbx
  unsigned int v8; // edi
  __int64 *v9; // rcx
  __int64 v10; // rdx
  __int64 v11; // rcx
  int v12; // eax
  unsigned __int64 v13; // r8
  __int64 v14; // r14
  __int64 v15; // rdx
  __int64 v16; // rax
  bool v17; // zf

  v7 = *(_QWORD *)a1;
  v8 = *(_DWORD *)(*(_QWORD *)a1 + 84LL);
  if ( v8 )
  {
    v9 = (__int64 *)(v7 + 40);
    v10 = 0;
    do
    {
      if ( (*(_DWORD *)((*v9 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v9 >> 1) & 3) > (*(_DWORD *)((a2 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                            | (unsigned int)(a2 >> 1)
                                                                                            & 3) )
        break;
      v10 = (unsigned int)(v10 + 1);
      ++v9;
    }
    while ( v8 != (_DWORD)v10 );
  }
  else
  {
    v10 = 0;
  }
  v11 = *(unsigned int *)(v7 + 80);
  v12 = *(_DWORD *)(a1 + 20);
  v13 = a1 + 8;
  *(_DWORD *)(a1 + 16) = 0;
  if ( (_DWORD)v11 )
    v7 += 8;
  v14 = (v10 << 32) | v8;
  v15 = 0;
  if ( !v12 )
  {
    sub_16CD150(a1 + 8, (const void *)(a1 + 24), 0, 16, v13, a6);
    v15 = 16LL * *(unsigned int *)(a1 + 16);
  }
  v16 = *(_QWORD *)(a1 + 8);
  *(_QWORD *)(v16 + v15) = v7;
  *(_QWORD *)(v16 + v15 + 8) = v14;
  v17 = (*(_DWORD *)(a1 + 16))++ == -1;
  if ( !v17 && *(_DWORD *)(*(_QWORD *)(a1 + 8) + 12LL) < *(_DWORD *)(*(_QWORD *)(a1 + 8) + 8LL) )
    sub_1DAAA30(a1, a2, v15, v11, v13);
}
