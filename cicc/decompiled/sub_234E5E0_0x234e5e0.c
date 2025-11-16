// Function: sub_234E5E0
// Address: 0x234e5e0
//
__int64 __fastcall sub_234E5E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v7; // rdi
  unsigned int v9; // r13d
  unsigned int v10; // r13d
  __int64 v11; // rax
  __int64 result; // rax
  const void *v13; // rax
  const void *v14; // rsi
  size_t v15; // rdx
  __int64 v16; // rdx
  size_t v17; // rdx
  int v18; // edx
  int v19; // eax

  v7 = (_QWORD *)(a1 + 32);
  *(v7 - 4) = *(_QWORD *)a2;
  *(v7 - 3) = *(_QWORD *)(a2 + 8);
  *(_QWORD *)(a1 + 16) = v7;
  *(_QWORD *)(a1 + 24) = 0x400000000LL;
  v9 = *(_DWORD *)(a2 + 24);
  if ( v9 )
  {
    a5 = a1 + 16;
    if ( a1 + 16 != a2 + 16 )
    {
      v13 = *(const void **)(a2 + 16);
      v14 = (const void *)(a2 + 32);
      if ( v13 == v14 )
      {
        v15 = 8LL * v9;
        if ( v9 <= 4
          || (sub_C8D5F0(a1 + 16, v7, v9, 8u, a5, v9),
              v7 = *(_QWORD **)(a1 + 16),
              v14 = *(const void **)(a2 + 16),
              (v15 = 8LL * *(unsigned int *)(a2 + 24)) != 0) )
        {
          memcpy(v7, v14, v15);
        }
        *(_DWORD *)(a1 + 24) = v9;
        *(_DWORD *)(a2 + 24) = 0;
      }
      else
      {
        *(_QWORD *)(a1 + 16) = v13;
        v19 = *(_DWORD *)(a2 + 28);
        *(_DWORD *)(a1 + 24) = v9;
        *(_DWORD *)(a1 + 28) = v19;
        *(_QWORD *)(a2 + 16) = v14;
        *(_QWORD *)(a2 + 24) = 0;
      }
    }
  }
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 64) = a1 + 80;
  v10 = *(_DWORD *)(a2 + 72);
  if ( v10 && a1 + 64 != a2 + 64 )
  {
    v16 = *(_QWORD *)(a2 + 64);
    if ( v16 == a2 + 80 )
    {
      sub_C8D5F0(a1 + 64, (const void *)(a1 + 80), v10, 0x10u, a5, a6);
      v17 = 16LL * *(unsigned int *)(a2 + 72);
      if ( v17 )
        memcpy(*(void **)(a1 + 64), *(const void **)(a2 + 64), v17);
      *(_DWORD *)(a1 + 72) = v10;
    }
    else
    {
      *(_QWORD *)(a1 + 64) = v16;
      v18 = *(_DWORD *)(a2 + 76);
      *(_DWORD *)(a1 + 72) = v10;
      *(_DWORD *)(a1 + 76) = v18;
      *(_QWORD *)(a2 + 64) = a2 + 80;
      *(_DWORD *)(a2 + 76) = 0;
    }
  }
  v11 = *(_QWORD *)(a2 + 80);
  *(_QWORD *)(a2 + 8) = 0;
  *(_QWORD *)a2 = 0;
  *(_QWORD *)(a1 + 80) = v11;
  result = *(_QWORD *)(a2 + 88);
  *(_QWORD *)(a2 + 80) = 0;
  *(_QWORD *)(a1 + 88) = result;
  *(_DWORD *)(a2 + 24) = 0;
  *(_DWORD *)(a2 + 72) = 0;
  return result;
}
