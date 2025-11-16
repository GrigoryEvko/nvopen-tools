// Function: sub_2FE0590
// Address: 0x2fe0590
//
__int64 __fastcall sub_2FE0590(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v5; // r9
  __int64 v6; // rdx
  unsigned int v7; // r12d
  unsigned __int64 v8; // rcx
  unsigned __int64 v9; // r8
  unsigned __int64 v10; // rcx
  __int64 v11; // rax
  __int64 v12; // rdx
  unsigned __int64 v13; // rcx
  unsigned __int8 v14; // [rsp+Fh] [rbp-21h]
  unsigned __int8 v15; // [rsp+Fh] [rbp-21h]
  _BYTE v16[17]; // [rsp+1Fh] [rbp-11h] BYREF

  result = sub_2FE04D0(a1, a2, (__int64)v16);
  if ( (_BYTE)result )
  {
    v6 = *(unsigned int *)(a3 + 8);
    v7 = v16[0];
    v8 = *(unsigned int *)(a3 + 12);
    v9 = v6 + 1;
    if ( v16[0] )
    {
      if ( v9 > v8 )
      {
        sub_C8D5F0(a3, (const void *)(a3 + 16), v6 + 1, 4u, v9, v5);
        v6 = *(unsigned int *)(a3 + 8);
      }
      *(_DWORD *)(*(_QWORD *)a3 + 4 * v6) = 1;
      v10 = *(unsigned int *)(a3 + 12);
      v11 = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
      *(_DWORD *)(a3 + 8) = v11;
      if ( v11 + 1 > v10 )
      {
        sub_C8D5F0(a3, (const void *)(a3 + 16), v11 + 1, 4u, v9, v5);
        v11 = *(unsigned int *)(a3 + 8);
      }
      *(_DWORD *)(*(_QWORD *)a3 + 4 * v11) = 3;
      ++*(_DWORD *)(a3 + 8);
      return v7;
    }
    else
    {
      if ( v9 > v8 )
      {
        v14 = result;
        sub_C8D5F0(a3, (const void *)(a3 + 16), v6 + 1, 4u, v9, v5);
        v6 = *(unsigned int *)(a3 + 8);
        result = v14;
      }
      *(_DWORD *)(*(_QWORD *)a3 + 4 * v6) = 0;
      v12 = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
      v13 = *(unsigned int *)(a3 + 12);
      *(_DWORD *)(a3 + 8) = v12;
      if ( v12 + 1 > v13 )
      {
        v15 = result;
        sub_C8D5F0(a3, (const void *)(a3 + 16), v12 + 1, 4u, v12 + 1, v5);
        v12 = *(unsigned int *)(a3 + 8);
        result = v15;
      }
      *(_DWORD *)(*(_QWORD *)a3 + 4 * v12) = 2;
      ++*(_DWORD *)(a3 + 8);
    }
  }
  return result;
}
