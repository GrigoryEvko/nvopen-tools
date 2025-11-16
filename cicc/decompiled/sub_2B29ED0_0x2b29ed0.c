// Function: sub_2B29ED0
// Address: 0x2b29ed0
//
__int64 __fastcall sub_2B29ED0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v7; // r13d
  void *v8; // rdi
  __int64 v9; // r8
  unsigned int v10; // ecx
  unsigned int v11; // edi
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // r9
  int v14; // edx

  v7 = *(_DWORD *)(a2 + 120);
  if ( !v7 )
    v7 = *(_DWORD *)(a2 + 8);
  v8 = (void *)(a1 + 16);
  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0xC00000000LL;
  if ( v7 > 0xC )
  {
    sub_C8D5F0(a1, (const void *)(a1 + 16), v7, 4u, a5, a6);
    memset(*(void **)a1, 255, 4LL * v7);
  }
  else if ( v7 )
  {
    memset(v8, 255, (size_t)v8 + 4 * v7 - a1 - 16);
  }
  *(_DWORD *)(a1 + 8) = v7;
  v9 = *(_QWORD *)(a2 + 144);
  v10 = *(_DWORD *)(*(_QWORD *)(a2 + 208) + 8LL * *(unsigned int *)(a2 + 216) - 4);
  v11 = *(_DWORD *)(a2 + 8) - v10;
  if ( v11 < v10 )
    v11 = *(_DWORD *)(*(_QWORD *)(a2 + 208) + 8LL * *(unsigned int *)(a2 + 216) - 4);
  if ( v9 != v9 + 4LL * *(unsigned int *)(a2 + 152) )
  {
    v12 = 0;
    v13 = (4 * (unsigned __int64)*(unsigned int *)(a2 + 152) - 4) >> 2;
    while ( 1 )
    {
      v14 = v12 + v11 - v10;
      if ( v10 > v12 )
        v14 = v12;
      *(_DWORD *)(*(_QWORD *)a1 + 4LL * *(unsigned int *)(v9 + 4 * v12)) = v14;
      if ( v12 == v13 )
        break;
      ++v12;
      v10 = *(_DWORD *)(*(_QWORD *)(a2 + 208) + 8LL * *(unsigned int *)(a2 + 216) - 4);
    }
  }
  return a1;
}
