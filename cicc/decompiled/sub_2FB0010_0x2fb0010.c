// Function: sub_2FB0010
// Address: 0x2fb0010
//
__int64 __fastcall sub_2FB0010(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rbx
  int v7; // eax
  int v8; // ecx
  unsigned __int64 v9; // rcx
  unsigned __int64 v10; // r8
  __int64 result; // rax
  __int64 v12; // r12

  *(_QWORD *)(a1 + 32) = a2;
  *(_DWORD *)(a1 + 96) = 0;
  *(_DWORD *)(a1 + 232) = 0;
  *(_DWORD *)(a2 + 64) = 0;
  *(_DWORD *)(a2 + 8) = 0;
  v6 = *(_QWORD *)(a1 + 32);
  v7 = *(_DWORD *)(*(_QWORD *)(a1 + 8) + 56LL);
  v8 = *(_DWORD *)(v6 + 64) & 0x3F;
  if ( v8 )
    *(_QWORD *)(*(_QWORD *)v6 + 8LL * *(unsigned int *)(v6 + 8) - 8) &= ~(-1LL << v8);
  v9 = *(unsigned int *)(v6 + 8);
  *(_DWORD *)(v6 + 64) = v7;
  v10 = (unsigned int)(v7 + 63) >> 6;
  if ( v10 != v9 )
  {
    if ( v10 >= v9 )
    {
      v12 = v10 - v9;
      if ( v10 > *(unsigned int *)(v6 + 12) )
      {
        sub_C8D5F0(v6, (const void *)(v6 + 16), v10, 8u, v10, a6);
        v9 = *(unsigned int *)(v6 + 8);
      }
      if ( 8 * v12 )
      {
        memset((void *)(*(_QWORD *)v6 + 8 * v9), 0, 8 * v12);
        LODWORD(v9) = *(_DWORD *)(v6 + 8);
      }
      v7 = *(_DWORD *)(v6 + 64);
      *(_DWORD *)(v6 + 8) = v12 + v9;
    }
    else
    {
      *(_DWORD *)(v6 + 8) = (unsigned int)(v7 + 63) >> 6;
    }
  }
  result = v7 & 0x3F;
  if ( (_DWORD)result )
    *(_QWORD *)(*(_QWORD *)v6 + 8LL * *(unsigned int *)(v6 + 8) - 8) &= ~(-1LL << result);
  return result;
}
