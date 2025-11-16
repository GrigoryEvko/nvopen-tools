// Function: sub_D25660
// Address: 0xd25660
//
__int64 __fastcall sub_D25660(__int64 a1, __int64 a2, __int64 a3, unsigned __int8 a4)
{
  __int64 v6; // rax
  __int64 result; // rax
  __int64 v8; // r9
  unsigned __int64 v9; // rcx
  unsigned __int64 v10; // r12
  unsigned __int64 v11; // rdx
  int v12; // eax
  unsigned __int64 *v13; // rdx
  __int64 v14; // [rsp+0h] [rbp-60h] BYREF
  __int64 v15; // [rsp+8h] [rbp-58h] BYREF
  _BYTE v16[80]; // [rsp+10h] [rbp-50h] BYREF

  v6 = *(unsigned int *)(a1 + 8);
  v14 = a3;
  v15 = v6;
  result = sub_D25400((__int64)v16, a2, &v14, &v15);
  if ( v16[32] )
  {
    v9 = *(unsigned int *)(a1 + 12);
    v10 = (4LL * a4) | a3 & 0xFFFFFFFFFFFFFFFBLL;
    v11 = *(unsigned int *)(a1 + 8);
    v12 = *(_DWORD *)(a1 + 8);
    if ( v11 >= v9 )
    {
      if ( v9 < v11 + 1 )
      {
        sub_C8D5F0(a1, (const void *)(a1 + 16), v11 + 1, 8u, v11 + 1, v8);
        v11 = *(unsigned int *)(a1 + 8);
      }
      result = *(_QWORD *)a1;
      *(_QWORD *)(*(_QWORD *)a1 + 8 * v11) = v10;
      ++*(_DWORD *)(a1 + 8);
    }
    else
    {
      v13 = (unsigned __int64 *)(*(_QWORD *)a1 + 8 * v11);
      if ( v13 )
      {
        *v13 = v10;
        v12 = *(_DWORD *)(a1 + 8);
      }
      result = (unsigned int)(v12 + 1);
      *(_DWORD *)(a1 + 8) = result;
    }
  }
  return result;
}
