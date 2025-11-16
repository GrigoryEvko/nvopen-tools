// Function: sub_205AAF0
// Address: 0x205aaf0
//
__int64 __fastcall sub_205AAF0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  void *v7; // rdi
  unsigned int v8; // r13d
  void *v9; // rdi
  unsigned int v10; // r13d
  __int64 v11; // rdx
  __int64 result; // rax
  size_t v13; // rdx
  size_t v14; // rdx

  v7 = (void *)(a1 + 16);
  *(_QWORD *)a1 = v7;
  *(_QWORD *)(a1 + 8) = 0x400000000LL;
  v8 = *(_DWORD *)(a2 + 8);
  if ( v8 && a1 != a2 )
  {
    v13 = 16LL * v8;
    if ( v8 <= 4
      || (sub_16CD150(a1, v7, v8, 16, v8, a6), v7 = *(void **)a1, (v13 = 16LL * *(unsigned int *)(a2 + 8)) != 0) )
    {
      memcpy(v7, *(const void **)a2, v13);
    }
    *(_DWORD *)(a1 + 8) = v8;
  }
  v9 = (void *)(a1 + 96);
  *(_QWORD *)(a1 + 80) = a1 + 96;
  *(_QWORD *)(a1 + 88) = 0x400000000LL;
  v10 = *(_DWORD *)(a2 + 88);
  if ( v10 )
  {
    a5 = a1 + 80;
    if ( a1 + 80 != a2 + 80 )
    {
      v14 = v10;
      if ( v10 <= 4
        || (sub_16CD150(a1 + 80, (const void *)(a1 + 96), v10, 1, a5, a6),
            v14 = *(unsigned int *)(a2 + 88),
            v9 = *(void **)(a1 + 80),
            *(_DWORD *)(a2 + 88)) )
      {
        memcpy(v9, *(const void **)(a2 + 80), v14);
      }
      *(_DWORD *)(a1 + 88) = v10;
    }
  }
  *(_QWORD *)(a1 + 104) = a1 + 120;
  *(_QWORD *)(a1 + 112) = 0x400000000LL;
  v11 = *(unsigned int *)(a2 + 112);
  if ( (_DWORD)v11 )
    sub_2045020(a1 + 104, a2 + 104, v11, a4, a5, a6);
  *(_QWORD *)(a1 + 136) = a1 + 152;
  *(_QWORD *)(a1 + 144) = 0x400000000LL;
  if ( *(_DWORD *)(a2 + 144) )
    sub_2045020(a1 + 136, a2 + 136, v11, a4, a5, a6);
  result = *(unsigned __int8 *)(a2 + 172);
  *(_BYTE *)(a1 + 172) = result;
  if ( (_BYTE)result )
  {
    result = *(unsigned int *)(a2 + 168);
    *(_DWORD *)(a1 + 168) = result;
  }
  return result;
}
