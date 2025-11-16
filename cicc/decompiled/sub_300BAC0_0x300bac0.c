// Function: sub_300BAC0
// Address: 0x300bac0
//
unsigned __int64 __fastcall sub_300BAC0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v6; // r12
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // rdx
  unsigned __int64 result; // rax
  int v10; // r13d
  unsigned __int64 v11; // r14
  unsigned __int64 v12; // rdx
  int v13; // r14d
  __int64 v14; // r15
  _DWORD *v15; // rax
  _DWORD *v16; // rcx
  int v17; // r14d
  __int64 v18; // r15
  _DWORD *v19; // rax
  __int64 v20; // rdx

  v6 = *(unsigned int *)(*(_QWORD *)(*(_QWORD *)(a1 + 24) + 32LL) + 64LL);
  v7 = *(unsigned int *)(a1 + 40);
  if ( v6 != v7 )
  {
    if ( v6 >= v7 )
    {
      v17 = *(_DWORD *)(a1 + 48);
      v18 = v6 - v7;
      if ( v6 > *(unsigned int *)(a1 + 44) )
      {
        sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), v6, 4u, a5, a6);
        v7 = *(unsigned int *)(a1 + 40);
      }
      v19 = (_DWORD *)(*(_QWORD *)(a1 + 32) + 4 * v7);
      v20 = v18;
      do
      {
        if ( v19 )
          *v19 = v17;
        ++v19;
        --v20;
      }
      while ( v20 );
      *(_DWORD *)(a1 + 40) += v18;
    }
    else
    {
      *(_DWORD *)(a1 + 40) = v6;
    }
  }
  v8 = *(unsigned int *)(a1 + 64);
  if ( v6 != v8 )
  {
    if ( v6 >= v8 )
    {
      v13 = *(_DWORD *)(a1 + 72);
      v14 = v6 - v8;
      if ( v6 > *(unsigned int *)(a1 + 68) )
      {
        sub_C8D5F0(a1 + 56, (const void *)(a1 + 72), v6, 4u, a5, a6);
        v8 = *(unsigned int *)(a1 + 64);
      }
      v15 = (_DWORD *)(*(_QWORD *)(a1 + 56) + 4 * v8);
      v16 = &v15[v14];
      if ( v15 != v16 )
      {
        do
          *v15++ = v13;
        while ( v16 != v15 );
        LODWORD(v8) = *(_DWORD *)(a1 + 64);
      }
      *(_DWORD *)(a1 + 64) = v14 + v8;
    }
    else
    {
      *(_DWORD *)(a1 + 64) = v6;
    }
  }
  result = *(unsigned int *)(a1 + 88);
  if ( v6 != result )
  {
    if ( v6 >= result )
    {
      v10 = *(_DWORD *)(a1 + 96);
      v11 = v6 - result;
      if ( v6 > *(unsigned int *)(a1 + 92) )
      {
        sub_C8D5F0(a1 + 80, (const void *)(a1 + 96), v6, 4u, a5, a6);
        result = *(unsigned int *)(a1 + 88);
      }
      result = *(_QWORD *)(a1 + 80) + 4 * result;
      v12 = v11;
      do
      {
        if ( result )
          *(_DWORD *)result = v10;
        result += 4LL;
        --v12;
      }
      while ( v12 );
      *(_DWORD *)(a1 + 88) += v11;
    }
    else
    {
      *(_DWORD *)(a1 + 88) = v6;
    }
  }
  return result;
}
