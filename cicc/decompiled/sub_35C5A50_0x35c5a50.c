// Function: sub_35C5A50
// Address: 0x35c5a50
//
__int64 __fastcall sub_35C5A50(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 (*v4)(void); // rdx
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // r9
  void *v8; // rdi
  __int64 v9; // r13
  __int64 v10; // rax
  unsigned __int64 v11; // rax
  int v12; // edx
  int v13; // ecx
  unsigned __int64 v14; // r8
  int v15; // edx
  __int64 v16; // rdx
  __int64 result; // rax
  __int64 i; // rdx
  __int64 v19; // r13

  v2 = *(_QWORD *)(a2 + 32);
  v4 = *(__int64 (**)(void))(**(_QWORD **)(v2 + 16) + 128LL);
  v5 = 0;
  if ( v4 != sub_2DAC790 )
    v5 = v4();
  *(_QWORD *)(a1 + 8) = v5;
  v6 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(v2 + 16) + 200LL))(*(_QWORD *)(v2 + 16));
  v8 = *(void **)(a1 + 96);
  v9 = v6;
  *(_QWORD *)a1 = v6;
  v10 = *(_QWORD *)(v2 + 32);
  *(_QWORD *)(a1 + 88) = v9;
  *(_QWORD *)(a1 + 16) = v10;
  v11 = *(unsigned int *)(a1 + 104);
  if ( 8 * v11 )
  {
    memset(v8, 0, 8 * v11);
    v11 = *(unsigned int *)(a1 + 104);
  }
  v12 = *(_DWORD *)(v9 + 44);
  v13 = *(_DWORD *)(a1 + 160) & 0x3F;
  if ( v13 )
  {
    *(_QWORD *)(*(_QWORD *)(a1 + 96) + 8 * v11 - 8) &= ~(-1LL << v13);
    v11 = *(unsigned int *)(a1 + 104);
  }
  *(_DWORD *)(a1 + 160) = v12;
  v14 = (unsigned int)(v12 + 63) >> 6;
  if ( v14 != v11 )
  {
    if ( v14 >= v11 )
    {
      v19 = v14 - v11;
      if ( v14 > *(unsigned int *)(a1 + 108) )
      {
        sub_C8D5F0(a1 + 96, (const void *)(a1 + 112), v14, 8u, v14, v7);
        v11 = *(unsigned int *)(a1 + 104);
      }
      if ( 8 * v19 )
      {
        memset((void *)(*(_QWORD *)(a1 + 96) + 8 * v11), 0, 8 * v19);
        LODWORD(v11) = *(_DWORD *)(a1 + 104);
      }
      v12 = *(_DWORD *)(a1 + 160);
      *(_DWORD *)(a1 + 104) = v19 + v11;
    }
    else
    {
      *(_DWORD *)(a1 + 104) = (unsigned int)(v12 + 63) >> 6;
    }
  }
  v15 = v12 & 0x3F;
  if ( v15 )
    *(_QWORD *)(*(_QWORD *)(a1 + 96) + 8LL * *(unsigned int *)(a1 + 104) - 8) &= ~(-1LL << v15);
  v16 = *(unsigned int *)(a1 + 48);
  result = *(_QWORD *)(a1 + 40);
  *(_QWORD *)(a1 + 24) = a2;
  for ( i = result + 16 * v16; i != result; *(_QWORD *)(result - 8) = 0 )
  {
    *(_DWORD *)(result + 4) = 0;
    result += 16;
  }
  return result;
}
