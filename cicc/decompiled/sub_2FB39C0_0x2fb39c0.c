// Function: sub_2FB39C0
// Address: 0x2fb39c0
//
__int64 __fastcall sub_2FB39C0(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rdi
  __int64 v8; // r13
  int v9; // edx
  int v10; // ecx
  unsigned int v11; // eax
  __int64 v12; // r12
  __int64 result; // rax
  unsigned __int64 v14; // r12
  __int64 v15; // rdx

  v7 = a1 + 8;
  v8 = *(_QWORD *)(v7 - 8);
  v9 = *(_DWORD *)(a1 + 20);
  v10 = *(_DWORD *)(v8 + 184);
  v11 = *(_DWORD *)(v8 + 188);
  *(_DWORD *)(a1 + 16) = 0;
  if ( v10 )
    v8 += 8;
  v12 = v11;
  result = 0;
  v14 = ((unsigned __int64)a2 << 32) | v12;
  if ( !v9 )
  {
    sub_C8D5F0(v7, (const void *)(a1 + 24), 1u, 0x10u, a5, a6);
    result = 16LL * *(unsigned int *)(a1 + 16);
  }
  v15 = *(_QWORD *)(a1 + 8);
  *(_QWORD *)(v15 + result) = v8;
  *(_QWORD *)(v15 + result + 8) = v14;
  ++*(_DWORD *)(a1 + 16);
  return result;
}
