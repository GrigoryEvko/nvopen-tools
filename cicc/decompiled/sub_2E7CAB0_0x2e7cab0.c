// Function: sub_2E7CAB0
// Address: 0x2e7cab0
//
__int64 __fastcall sub_2E7CAB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rdx
  void *v9; // rdi
  unsigned int v10; // r13d
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 result; // rax
  __int64 v14; // rax
  const void *v15; // rsi
  size_t v16; // rdx
  int v17; // eax

  *(_QWORD *)a1 = *(_QWORD *)a2;
  *(_QWORD *)(a1 + 8) = a1 + 24;
  *(_QWORD *)(a1 + 16) = 0x100000000LL;
  v8 = *(unsigned int *)(a2 + 16);
  if ( (_DWORD)v8 )
    sub_2E78330(a1 + 8, (char **)(a2 + 8), v8, a4, a5, a6);
  *(_QWORD *)(a1 + 32) = a1 + 48;
  *(_QWORD *)(a1 + 40) = 0x100000000LL;
  if ( *(_DWORD *)(a2 + 40) )
    sub_2E78330(a1 + 32, (char **)(a2 + 32), v8, a4, a5, a6);
  v9 = (void *)(a1 + 72);
  *(_QWORD *)(a1 + 56) = a1 + 72;
  *(_QWORD *)(a1 + 64) = 0x100000000LL;
  v10 = *(_DWORD *)(a2 + 64);
  if ( v10 && a1 + 56 != a2 + 56 )
  {
    v14 = *(_QWORD *)(a2 + 56);
    v15 = (const void *)(a2 + 72);
    if ( v14 == a2 + 72 )
    {
      v16 = 16;
      if ( v10 == 1
        || (sub_C8D5F0(a1 + 56, (const void *)(a1 + 72), v10, 0x10u, a1 + 56, v10),
            v9 = *(void **)(a1 + 56),
            v15 = *(const void **)(a2 + 56),
            (v16 = 16LL * *(unsigned int *)(a2 + 64)) != 0) )
      {
        memcpy(v9, v15, v16);
      }
      *(_DWORD *)(a1 + 64) = v10;
      *(_DWORD *)(a2 + 64) = 0;
    }
    else
    {
      *(_QWORD *)(a1 + 56) = v14;
      v17 = *(_DWORD *)(a2 + 68);
      *(_DWORD *)(a1 + 64) = v10;
      *(_DWORD *)(a1 + 68) = v17;
      *(_QWORD *)(a2 + 56) = v15;
      *(_QWORD *)(a2 + 64) = 0;
    }
  }
  *(_QWORD *)(a1 + 88) = *(_QWORD *)(a2 + 88);
  v11 = *(_QWORD *)(a2 + 96);
  *(_QWORD *)(a2 + 96) = 0;
  *(_QWORD *)(a1 + 96) = v11;
  v12 = *(_QWORD *)(a2 + 104);
  *(_QWORD *)(a2 + 104) = 0;
  *(_QWORD *)(a1 + 104) = v12;
  result = *(_QWORD *)(a2 + 112);
  *(_QWORD *)(a2 + 112) = 0;
  *(_QWORD *)(a1 + 112) = result;
  return result;
}
