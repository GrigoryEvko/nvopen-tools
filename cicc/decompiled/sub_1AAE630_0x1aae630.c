// Function: sub_1AAE630
// Address: 0x1aae630
//
__int64 __fastcall sub_1AAE630(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 v8; // rdi
  size_t v9; // r12
  const void *v10; // r15
  unsigned int v11; // r12d
  unsigned int v12; // edx
  __int64 v13; // rax
  __int64 result; // rax
  __int64 v15; // rdx

  v8 = *(unsigned int *)(a3 + 8);
  v9 = *(_QWORD *)(a2 + 8);
  v10 = *(const void **)a2;
  if ( v9 > (unsigned __int64)*(unsigned int *)(a3 + 12) - v8 )
  {
    sub_16CD150(a3, (const void *)(a3 + 16), v9 + v8, 1, a5, a6);
    v8 = *(unsigned int *)(a3 + 8);
  }
  if ( v9 )
  {
    memcpy((void *)(*(_QWORD *)a3 + v8), v10, v9);
    LODWORD(v8) = *(_DWORD *)(a3 + 8);
  }
  v11 = v8 + v9;
  v12 = *(_DWORD *)(a3 + 12);
  *(_DWORD *)(a3 + 8) = v11;
  v13 = v11;
  if ( *(_BYTE *)(*(_QWORD *)a1 + 8LL) == 2 )
  {
    if ( v12 <= v11 )
    {
      sub_16CD150(a3, (const void *)(a3 + 16), 0, 1, a5, a6);
      v13 = *(unsigned int *)(a3 + 8);
    }
    *(_BYTE *)(*(_QWORD *)a3 + v13) = 102;
    result = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
    *(_DWORD *)(a3 + 8) = result;
  }
  else
  {
    if ( v12 <= v11 )
    {
      sub_16CD150(a3, (const void *)(a3 + 16), 0, 1, a5, a6);
      v13 = *(unsigned int *)(a3 + 8);
    }
    *(_BYTE *)(*(_QWORD *)a3 + v13) = 108;
    result = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
    *(_DWORD *)(a3 + 8) = result;
  }
  v15 = *(_QWORD *)a3;
  *(_QWORD *)(a2 + 8) = result;
  *(_QWORD *)a2 = v15;
  return result;
}
