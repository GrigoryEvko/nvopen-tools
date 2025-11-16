// Function: sub_1E72840
// Address: 0x1e72840
//
unsigned __int64 __fastcall sub_1E72840(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  unsigned __int64 result; // rax
  int v10; // r8d
  int v11; // r9d
  unsigned __int64 v12; // r12
  unsigned __int64 v13; // rax
  int v14; // r13d
  __int64 v15; // rdx
  void *v16; // rdi
  __int64 v17; // rdx
  _DWORD *v18; // rax
  _DWORD *i; // rdx
  __int64 v20; // rax

  sub_1E72570(a1, a2, a3, a4, a5, a6);
  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 8) = a3;
  *(_QWORD *)(a1 + 16) = a4;
  result = sub_1F4B670(a3);
  if ( (_BYTE)result )
  {
    v12 = *(unsigned int *)(*(_QWORD *)(a1 + 8) + 48LL);
    v13 = *(unsigned int *)(a1 + 200);
    v14 = *(_DWORD *)(*(_QWORD *)(a1 + 8) + 48LL);
    if ( v12 >= v13 )
    {
      if ( v12 > v13 )
      {
        if ( v12 > *(unsigned int *)(a1 + 204) )
        {
          sub_16CD150(a1 + 192, (const void *)(a1 + 208), *(unsigned int *)(*(_QWORD *)(a1 + 8) + 48LL), 4, v10, v11);
          v13 = *(unsigned int *)(a1 + 200);
        }
        v17 = *(_QWORD *)(a1 + 192);
        v18 = (_DWORD *)(v17 + 4 * v13);
        for ( i = (_DWORD *)(v17 + 4 * v12); i != v18; ++v18 )
        {
          if ( v18 )
            *v18 = 0;
        }
        v20 = *(_QWORD *)(a1 + 8);
        *(_DWORD *)(a1 + 200) = v12;
        v12 = *(unsigned int *)(v20 + 48);
        v14 = *(_DWORD *)(v20 + 48);
      }
    }
    else
    {
      *(_DWORD *)(a1 + 200) = v12;
    }
    result = *(unsigned int *)(a1 + 296);
    if ( result <= v12 )
    {
      if ( result >= v12 )
        return result;
      if ( *(unsigned int *)(a1 + 300) < v12 )
      {
        sub_16CD150(a1 + 288, (const void *)(a1 + 304), v12, 4, v10, v11);
        result = *(unsigned int *)(a1 + 296);
      }
      v15 = *(_QWORD *)(a1 + 288);
      v16 = (void *)(v15 + 4 * result);
      if ( v16 != (void *)(v15 + 4 * v12) )
        result = (unsigned __int64)memset(v16, 255, 4 * (v12 - result));
    }
    *(_DWORD *)(a1 + 296) = v14;
  }
  return result;
}
