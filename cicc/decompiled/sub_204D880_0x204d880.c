// Function: sub_204D880
// Address: 0x204d880
//
__int64 __fastcall sub_204D880(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  unsigned __int64 v7; // r13
  __int64 v8; // rdi
  const void *v9; // r15
  size_t v10; // r14
  __int64 v11; // rdx
  __int64 v12; // rdi
  size_t v13; // r14
  const void *v14; // r15
  int v15; // eax
  __int64 v16; // rdx
  unsigned __int64 v17; // r13
  const void *v18; // r15
  __int64 result; // rax
  int v20; // r12d

  v7 = *(unsigned int *)(a2 + 8);
  v8 = *(unsigned int *)(a1 + 8);
  v9 = *(const void **)a2;
  v10 = 16 * v7;
  if ( v7 > (unsigned __int64)*(unsigned int *)(a1 + 12) - v8 )
  {
    sub_16CD150(a1, (const void *)(a1 + 16), v7 + v8, 16, a5, a6);
    v8 = *(unsigned int *)(a1 + 8);
  }
  if ( v10 )
  {
    memcpy((void *)(*(_QWORD *)a1 + 16 * v8), v9, v10);
    LODWORD(v8) = *(_DWORD *)(a1 + 8);
  }
  v11 = *(unsigned int *)(a1 + 92);
  *(_DWORD *)(a1 + 8) = v7 + v8;
  v12 = *(unsigned int *)(a1 + 88);
  v13 = *(unsigned int *)(a2 + 88);
  v14 = *(const void **)(a2 + 80);
  v15 = *(_DWORD *)(a1 + 88);
  if ( v13 > v11 - v12 )
  {
    sub_16CD150(a1 + 80, (const void *)(a1 + 96), v13 + v12, 1, a5, a6);
    v12 = *(unsigned int *)(a1 + 88);
    v15 = *(_DWORD *)(a1 + 88);
  }
  if ( (_DWORD)v13 )
  {
    memcpy((void *)(*(_QWORD *)(a1 + 80) + v12), v14, v13);
    v15 = *(_DWORD *)(a1 + 88);
  }
  v16 = *(unsigned int *)(a1 + 112);
  *(_DWORD *)(a1 + 88) = v13 + v15;
  v17 = *(unsigned int *)(a2 + 112);
  v18 = *(const void **)(a2 + 104);
  if ( v17 > (unsigned __int64)*(unsigned int *)(a1 + 116) - v16 )
  {
    sub_16CD150(a1 + 104, (const void *)(a1 + 120), v17 + v16, 4, a5, a6);
    v16 = *(unsigned int *)(a1 + 112);
  }
  if ( 4 * v17 )
  {
    memcpy((void *)(*(_QWORD *)(a1 + 104) + 4 * v16), v18, 4 * v17);
    LODWORD(v16) = *(_DWORD *)(a1 + 112);
  }
  result = *(unsigned int *)(a1 + 144);
  *(_DWORD *)(a1 + 112) = v17 + v16;
  v20 = *(_DWORD *)(a2 + 112);
  if ( (unsigned int)result >= *(_DWORD *)(a1 + 148) )
  {
    sub_16CD150(a1 + 136, (const void *)(a1 + 152), 0, 4, a5, a6);
    result = *(unsigned int *)(a1 + 144);
  }
  *(_DWORD *)(*(_QWORD *)(a1 + 136) + 4 * result) = v20;
  ++*(_DWORD *)(a1 + 144);
  return result;
}
