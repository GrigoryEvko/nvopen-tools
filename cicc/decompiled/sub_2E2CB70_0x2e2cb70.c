// Function: sub_2E2CB70
// Address: 0x2e2cb70
//
__int64 __fastcall sub_2E2CB70(_QWORD *a1, __int64 a2, unsigned int a3, __int64 *a4, char a5, unsigned __int8 *a6)
{
  __int64 v9; // r8
  unsigned __int8 v10; // cl
  char *v11; // rax
  __int64 v12; // r13
  __int64 v13; // rax
  _QWORD *v14; // rax
  __int64 v15; // rcx
  __int64 v16; // rax
  __int64 result; // rax
  char v18; // [rsp+1Fh] [rbp-31h] BYREF

  if ( a5 )
    *a4 += *(_QWORD *)(*(_QWORD *)(a2 + 8) + 40LL * (*(_DWORD *)(a2 + 32) + a3) + 8);
  v9 = a3;
  v10 = *(_BYTE *)(*(_QWORD *)(a2 + 8) + 40LL * (*(_DWORD *)(a2 + 32) + a3) + 16);
  v11 = &v18;
  if ( v10 <= *a6 )
    v11 = (char *)a6;
  v18 = *(_BYTE *)(*(_QWORD *)(a2 + 8) + 40LL * (*(_DWORD *)(a2 + 32) + a3) + 16);
  *a6 = *v11;
  v12 = -(1LL << v10) & ((1LL << v10) + *a4 - 1);
  *a4 = v12;
  if ( a5 )
    v12 = -v12;
  *(_QWORD *)(*a1 + 8LL * (int)a3) = v12;
  v13 = *(unsigned int *)(a2 + 136);
  if ( v13 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 140) )
  {
    sub_C8D5F0(a2 + 128, (const void *)(a2 + 144), v13 + 1, 0x10u, a3, (__int64)a6);
    v13 = *(unsigned int *)(a2 + 136);
    v9 = a3;
  }
  v14 = (_QWORD *)(*(_QWORD *)(a2 + 128) + 16 * v13);
  *v14 = v9;
  v14[1] = v12;
  v15 = *(_QWORD *)(a2 + 8);
  v16 = *(_DWORD *)(a2 + 32) + a3;
  ++*(_DWORD *)(a2 + 136);
  result = v15 + 40 * v16;
  *(_BYTE *)(result + 32) = 1;
  if ( !a5 )
  {
    result = *(_QWORD *)(*(_QWORD *)(a2 + 8) + 40LL * (*(_DWORD *)(a2 + 32) + a3) + 8);
    *a4 += result;
  }
  return result;
}
