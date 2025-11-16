// Function: sub_2A3B9E0
// Address: 0x2a3b9e0
//
__int64 __fastcall sub_2A3B9E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rdx
  void *v9; // rdi
  unsigned int v10; // r13d
  __int64 result; // rax
  void *v12; // rdi
  unsigned int v13; // r13d
  __int64 v14; // rax
  const void *v15; // rsi
  size_t v16; // rdx
  __int64 v17; // rax
  const void *v18; // rsi
  size_t v19; // rdx
  int v20; // eax

  *(_QWORD *)a1 = *(_QWORD *)a2;
  *(_QWORD *)(a1 + 8) = a1 + 24;
  *(_QWORD *)(a1 + 16) = 0x200000000LL;
  v8 = *(unsigned int *)(a2 + 16);
  if ( (_DWORD)v8 )
    sub_2A399F0(a1 + 8, (char **)(a2 + 8), v8, a4, a5, a6);
  *(_QWORD *)(a1 + 40) = a1 + 56;
  *(_QWORD *)(a1 + 48) = 0x200000000LL;
  if ( *(_DWORD *)(a2 + 48) )
    sub_2A399F0(a1 + 40, (char **)(a2 + 40), v8, a4, a5, a6);
  v9 = (void *)(a1 + 88);
  *(_QWORD *)(a1 + 72) = a1 + 88;
  *(_QWORD *)(a1 + 80) = 0x200000000LL;
  v10 = *(_DWORD *)(a2 + 80);
  if ( v10 && a1 + 72 != a2 + 72 )
  {
    v17 = *(_QWORD *)(a2 + 72);
    v18 = (const void *)(a2 + 88);
    if ( v17 == a2 + 88 )
    {
      v19 = 8LL * v10;
      if ( v10 <= 2
        || (sub_C8D5F0(a1 + 72, (const void *)(a1 + 88), v10, 8u, a1 + 72, v10),
            v9 = *(void **)(a1 + 72),
            v18 = *(const void **)(a2 + 72),
            (v19 = 8LL * *(unsigned int *)(a2 + 80)) != 0) )
      {
        memcpy(v9, v18, v19);
      }
      *(_DWORD *)(a1 + 80) = v10;
      *(_DWORD *)(a2 + 80) = 0;
    }
    else
    {
      *(_QWORD *)(a1 + 72) = v17;
      v20 = *(_DWORD *)(a2 + 84);
      *(_DWORD *)(a1 + 80) = v10;
      *(_DWORD *)(a1 + 84) = v20;
      *(_QWORD *)(a2 + 72) = v18;
      *(_QWORD *)(a2 + 80) = 0;
    }
  }
  result = 0x200000000LL;
  v12 = (void *)(a1 + 120);
  *(_QWORD *)(a1 + 104) = a1 + 120;
  *(_QWORD *)(a1 + 112) = 0x200000000LL;
  v13 = *(_DWORD *)(a2 + 112);
  if ( v13 )
  {
    result = a2 + 104;
    if ( a1 + 104 != a2 + 104 )
    {
      v14 = *(_QWORD *)(a2 + 104);
      v15 = (const void *)(a2 + 120);
      if ( v14 == a2 + 120 )
      {
        v16 = 8LL * v13;
        if ( v13 <= 2
          || (result = sub_C8D5F0(a1 + 104, (const void *)(a1 + 120), v13, 8u, a1 + 104, v13),
              v12 = *(void **)(a1 + 104),
              v15 = *(const void **)(a2 + 104),
              (v16 = 8LL * *(unsigned int *)(a2 + 112)) != 0) )
        {
          result = (__int64)memcpy(v12, v15, v16);
        }
        *(_DWORD *)(a1 + 112) = v13;
        *(_DWORD *)(a2 + 112) = 0;
      }
      else
      {
        *(_QWORD *)(a1 + 104) = v14;
        result = *(unsigned int *)(a2 + 116);
        *(_DWORD *)(a1 + 112) = v13;
        *(_DWORD *)(a1 + 116) = result;
        *(_QWORD *)(a2 + 104) = v15;
        *(_QWORD *)(a2 + 112) = 0;
      }
    }
  }
  return result;
}
