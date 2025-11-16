// Function: sub_23DEC70
// Address: 0x23dec70
//
__int64 __fastcall sub_23DEC70(__int64 a1, unsigned int **a2, unsigned __int64 a3, int a4, int a5, int a6, __int64 a7)
{
  __int64 result; // rax
  __int64 v8; // r9
  __int64 v9; // rdx
  __int64 v10; // [rsp+10h] [rbp-18h]

  result = sub_921880(a2, a3, a4, a5, a6, a7, 0);
  if ( *(_BYTE *)(a1 + 8) )
  {
    v9 = *(unsigned int *)(a1 + 24);
    if ( v9 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 28) )
    {
      v10 = result;
      sub_C8D5F0(a1 + 16, (const void *)(a1 + 32), v9 + 1, 8u, v9 + 1, v8);
      v9 = *(unsigned int *)(a1 + 24);
      result = v10;
    }
    *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8 * v9) = result;
    ++*(_DWORD *)(a1 + 24);
  }
  return result;
}
