// Function: sub_D25720
// Address: 0xd25720
//
__int64 __fastcall sub_D25720(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r13
  __int64 v6; // rsi
  __int64 v7; // rax
  __int64 result; // rax
  __int64 v9; // r9
  unsigned __int64 v10; // rdx
  unsigned __int64 v11; // rcx
  int v12; // eax
  unsigned __int64 *v13; // rdx
  unsigned __int64 v14; // r12
  __int64 v15; // [rsp+0h] [rbp-60h] BYREF
  __int64 v16; // [rsp+8h] [rbp-58h] BYREF
  _BYTE v17[80]; // [rsp+10h] [rbp-50h] BYREF

  v3 = a2 + 24;
  v6 = a2 + 72;
  v7 = *(unsigned int *)(v6 - 40);
  v15 = a3;
  v16 = v7;
  result = sub_D25400((__int64)v17, v6, &v15, &v16);
  if ( v17[32] )
  {
    v10 = *(unsigned int *)(a2 + 32);
    v11 = *(unsigned int *)(a2 + 36);
    v12 = *(_DWORD *)(a2 + 32);
    if ( v10 >= v11 )
    {
      v14 = a3 & 0xFFFFFFFFFFFFFFFBLL;
      if ( v11 < v10 + 1 )
      {
        sub_C8D5F0(v3, (const void *)(a2 + 40), v10 + 1, 8u, v10 + 1, v9);
        v10 = *(unsigned int *)(a2 + 32);
      }
      result = *(_QWORD *)(a2 + 24);
      *(_QWORD *)(result + 8 * v10) = v14;
      ++*(_DWORD *)(a2 + 32);
    }
    else
    {
      v13 = (unsigned __int64 *)(*(_QWORD *)(a2 + 24) + 8 * v10);
      if ( v13 )
      {
        *v13 = a3 & 0xFFFFFFFFFFFFFFFBLL;
        v12 = *(_DWORD *)(a2 + 32);
      }
      result = (unsigned int)(v12 + 1);
      *(_DWORD *)(a2 + 32) = result;
    }
  }
  return result;
}
