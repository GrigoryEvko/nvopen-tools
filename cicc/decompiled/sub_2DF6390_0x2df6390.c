// Function: sub_2DF6390
// Address: 0x2df6390
//
unsigned __int64 __fastcall sub_2DF6390(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
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
  unsigned __int64 result; // rax

  v7 = *(_QWORD *)a1;
  v8 = *(_DWORD *)(*(_QWORD *)a1 + 164LL);
  if ( v8 )
  {
    v9 = (__int64 *)(v7 + 80);
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
  v11 = *(unsigned int *)(v7 + 160);
  v12 = *(_DWORD *)(a1 + 20);
  v13 = a1 + 8;
  *(_DWORD *)(a1 + 16) = 0;
  if ( (_DWORD)v11 )
    v7 += 8;
  v14 = (v10 << 32) | v8;
  v15 = 0;
  if ( !v12 )
  {
    sub_C8D5F0(a1 + 8, (const void *)(a1 + 24), 1u, 0x10u, v13, a6);
    v15 = 16LL * *(unsigned int *)(a1 + 16);
  }
  result = *(_QWORD *)(a1 + 8);
  *(_QWORD *)(result + v15) = v7;
  *(_QWORD *)(result + v15 + 8) = v14;
  if ( (*(_DWORD *)(a1 + 16))++ != -1 )
  {
    result = *(_QWORD *)(a1 + 8);
    if ( *(_DWORD *)(result + 12) < *(_DWORD *)(result + 8) )
      return sub_2DF5F90(a1, a2, v15, v11, v13);
  }
  return result;
}
