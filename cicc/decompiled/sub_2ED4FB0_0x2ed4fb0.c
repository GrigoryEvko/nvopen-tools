// Function: sub_2ED4FB0
// Address: 0x2ed4fb0
//
unsigned __int64 __fastcall sub_2ED4FB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 result; // rax
  void *v8; // rdi
  int v9; // edx
  int v10; // ecx
  unsigned __int64 v11; // r8
  int v12; // edx
  unsigned __int64 v13; // r12

  *(_QWORD *)a1 = a2;
  result = *(unsigned int *)(a1 + 16);
  v8 = *(void **)(a1 + 8);
  if ( 8 * result )
  {
    memset(v8, 0, 8 * result);
    result = *(unsigned int *)(a1 + 16);
  }
  v9 = *(_DWORD *)(a2 + 44);
  v10 = *(_DWORD *)(a1 + 72) & 0x3F;
  if ( v10 )
  {
    *(_QWORD *)(*(_QWORD *)(a1 + 8) + 8 * result - 8) &= ~(-1LL << v10);
    result = *(unsigned int *)(a1 + 16);
  }
  *(_DWORD *)(a1 + 72) = v9;
  v11 = (unsigned int)(v9 + 63) >> 6;
  if ( v11 != result )
  {
    if ( v11 >= result )
    {
      v13 = v11 - result;
      if ( v11 > *(unsigned int *)(a1 + 20) )
      {
        sub_C8D5F0(a1 + 8, (const void *)(a1 + 24), v11, 8u, v11, a6);
        result = *(unsigned int *)(a1 + 16);
      }
      if ( 8 * v13 )
      {
        memset((void *)(*(_QWORD *)(a1 + 8) + 8 * result), 0, 8 * v13);
        result = *(unsigned int *)(a1 + 16);
      }
      result += v13;
      v9 = *(_DWORD *)(a1 + 72);
      *(_DWORD *)(a1 + 16) = result;
    }
    else
    {
      *(_DWORD *)(a1 + 16) = (unsigned int)(v9 + 63) >> 6;
    }
  }
  v12 = v9 & 0x3F;
  if ( v12 )
  {
    result = ~(-1LL << v12);
    *(_QWORD *)(*(_QWORD *)(a1 + 8) + 8LL * *(unsigned int *)(a1 + 16) - 8) &= result;
  }
  return result;
}
