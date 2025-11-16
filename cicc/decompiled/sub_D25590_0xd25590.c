// Function: sub_D25590
// Address: 0xd25590
//
__int64 __fastcall sub_D25590(__int64 a1, __int64 a2, unsigned __int8 a3)
{
  __int64 v5; // rax
  __int64 v6; // r8
  __int64 v7; // r9
  unsigned __int64 v8; // rcx
  unsigned __int64 v9; // rdi
  int v10; // eax
  unsigned __int64 *v11; // rdx
  __int64 result; // rax
  __int64 v13; // [rsp+0h] [rbp-60h] BYREF
  __int64 v14; // [rsp+8h] [rbp-58h] BYREF
  _BYTE v15[80]; // [rsp+10h] [rbp-50h] BYREF

  v5 = *(unsigned int *)(a1 + 8);
  v13 = a2;
  v14 = v5;
  sub_D25400((__int64)v15, a1 + 48, &v13, &v14);
  v8 = *(unsigned int *)(a1 + 8);
  v9 = *(unsigned int *)(a1 + 12);
  if ( v8 >= v9 )
  {
    if ( v9 < v8 + 1 )
    {
      sub_C8D5F0(a1, (const void *)(a1 + 16), v8 + 1, 8u, v6, v7);
      v8 = *(unsigned int *)(a1 + 8);
    }
    result = *(_QWORD *)a1;
    *(_QWORD *)(*(_QWORD *)a1 + 8 * v8) = (4LL * a3) | a2 & 0xFFFFFFFFFFFFFFFBLL;
    ++*(_DWORD *)(a1 + 8);
  }
  else
  {
    v10 = *(_DWORD *)(a1 + 8);
    v11 = (unsigned __int64 *)(*(_QWORD *)a1 + 8 * v8);
    if ( v11 )
    {
      *v11 = (4LL * a3) | a2 & 0xFFFFFFFFFFFFFFFBLL;
      v10 = *(_DWORD *)(a1 + 8);
    }
    result = (unsigned int)(v10 + 1);
    *(_DWORD *)(a1 + 8) = result;
  }
  return result;
}
