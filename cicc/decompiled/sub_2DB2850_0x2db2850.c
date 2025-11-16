// Function: sub_2DB2850
// Address: 0x2db2850
//
__int64 __fastcall sub_2DB2850(__int64 a1, __int64 a2)
{
  __int64 (*v3)(void); // rdx
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // r12
  __int64 v9; // rax
  unsigned int v10; // r13d
  __int64 v11; // rax
  unsigned __int64 v12; // rdi
  int v13; // ecx
  __int64 result; // rax
  unsigned int v15; // r12d
  __int64 v16; // r13
  __int64 v17; // rdi
  int v18; // ecx

  v3 = *(__int64 (**)(void))(**(_QWORD **)(a2 + 16) + 128LL);
  v4 = 0;
  if ( v3 != sub_2DAC790 )
    v4 = v3();
  *(_QWORD *)a1 = v4;
  v5 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a2 + 16) + 200LL))(*(_QWORD *)(a2 + 16));
  *(_QWORD *)(a1 + 8) = v5;
  v8 = v5;
  v9 = *(_QWORD *)(a2 + 32);
  *(_DWORD *)(a1 + 680) = 0;
  *(_QWORD *)(a1 + 16) = v9;
  v10 = *(_DWORD *)(v8 + 44);
  if ( v10 < *(_DWORD *)(a1 + 728) >> 2 || v10 > *(_DWORD *)(a1 + 728) )
  {
    v11 = (__int64)_libc_calloc(v10, 1u);
    if ( !v11 && (v10 || (v11 = malloc(1u)) == 0) )
      sub_C64F00("Allocation failed", 1u);
    v12 = *(_QWORD *)(a1 + 720);
    *(_QWORD *)(a1 + 720) = v11;
    if ( v12 )
    {
      _libc_free(v12);
      v8 = *(_QWORD *)(a1 + 8);
    }
    *(_DWORD *)(a1 + 728) = v10;
  }
  *(_DWORD *)(a1 + 664) = 0;
  *(_DWORD *)(a1 + 608) = 0;
  v13 = *(_DWORD *)(v8 + 44);
  *(_DWORD *)(a1 + 664) = v13;
  result = (unsigned int)(v13 + 63) >> 6;
  v15 = (unsigned int)(v13 + 63) >> 6;
  if ( v15 )
  {
    v16 = (unsigned int)result;
    v17 = 0;
    if ( *(_DWORD *)(a1 + 612) < (unsigned int)result )
    {
      sub_C8D5F0(a1 + 600, (const void *)(a1 + 616), (unsigned int)result, 8u, v6, v7);
      v17 = 8LL * *(unsigned int *)(a1 + 608);
    }
    result = (__int64)memset((void *)(*(_QWORD *)(a1 + 600) + v17), 0, 8 * v16);
    *(_DWORD *)(a1 + 608) += v15;
    v13 = *(_DWORD *)(a1 + 664);
  }
  v18 = v13 & 0x3F;
  if ( v18 )
  {
    result = ~(-1LL << v18);
    *(_QWORD *)(*(_QWORD *)(a1 + 600) + 8LL * *(unsigned int *)(a1 + 608) - 8) &= result;
  }
  return result;
}
