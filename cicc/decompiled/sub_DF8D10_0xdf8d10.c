// Function: sub_DF8D10
// Address: 0xdf8d10
//
size_t __fastcall sub_DF8D10(__int64 a1, int a2, __int64 a3, char *a4, __int64 a5)
{
  const void *v5; // r13
  char *v7; // rsi
  __int64 v8; // rdi
  size_t result; // rax
  __int64 v10; // r8
  __int64 v11; // r9
  unsigned __int64 v12; // rdx
  __int64 v13; // r12
  __int64 v14; // r15
  __int64 v15; // r14

  v5 = (const void *)(a1 + 40);
  *(_DWORD *)(a1 + 16) = a2;
  v7 = (char *)(a1 + 88);
  v8 = a1 + 72;
  *(_QWORD *)(v8 - 72) = 0;
  *(_QWORD *)(v8 - 64) = a3;
  *(_QWORD *)(v8 - 48) = v5;
  *(_QWORD *)(v8 - 40) = 0x400000000LL;
  *(_QWORD *)v8 = v7;
  *(_QWORD *)(v8 + 8) = 0x400000000LL;
  *(_DWORD *)(v8 + 48) = 0;
  *(_QWORD *)(v8 + 56) = 0;
  *(_DWORD *)(v8 + 64) = 1;
  *(_QWORD *)(v8 + 72) = 0;
  result = sub_DF6BA0(v8, v7, a4, &a4[8 * a5]);
  v12 = *(unsigned int *)(a1 + 80);
  if ( *(_DWORD *)(a1 + 36) < (unsigned int)v12 )
  {
    result = sub_C8D5F0(a1 + 24, v5, v12, 8u, v10, v11);
    v12 = *(unsigned int *)(a1 + 80);
  }
  v13 = *(_QWORD *)(a1 + 72);
  v14 = v13 + 8 * v12;
  if ( v13 != v14 )
  {
    result = *(unsigned int *)(a1 + 32);
    do
    {
      v15 = *(_QWORD *)(*(_QWORD *)v13 + 8LL);
      if ( result + 1 > *(unsigned int *)(a1 + 36) )
      {
        sub_C8D5F0(a1 + 24, v5, result + 1, 8u, v10, v11);
        result = *(unsigned int *)(a1 + 32);
      }
      v13 += 8;
      *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8 * result) = v15;
      result = (unsigned int)(*(_DWORD *)(a1 + 32) + 1);
      *(_DWORD *)(a1 + 32) = result;
    }
    while ( v14 != v13 );
  }
  return result;
}
