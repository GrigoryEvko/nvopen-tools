// Function: sub_1F3B880
// Address: 0x1f3b880
//
__int64 __fastcall sub_1F3B880(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  int v5; // r8d
  int v6; // r9d
  unsigned int v7; // r12d
  __int64 v8; // rdx
  unsigned int v9; // ecx
  __int64 v10; // rax
  __int64 v11; // rdx
  unsigned __int8 v12; // [rsp+Fh] [rbp-21h]
  unsigned __int8 v13; // [rsp+Fh] [rbp-21h]
  _BYTE v14[17]; // [rsp+1Fh] [rbp-11h] BYREF

  result = sub_1F3B7F0(a1, a2, v14);
  if ( (_BYTE)result )
  {
    v7 = v14[0];
    v8 = *(unsigned int *)(a3 + 8);
    v9 = *(_DWORD *)(a3 + 12);
    if ( v14[0] )
    {
      if ( (unsigned int)v8 >= v9 )
      {
        sub_16CD150(a3, (const void *)(a3 + 16), 0, 4, v5, v6);
        v8 = *(unsigned int *)(a3 + 8);
      }
      *(_DWORD *)(*(_QWORD *)a3 + 4 * v8) = 1;
      v10 = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
      *(_DWORD *)(a3 + 8) = v10;
      if ( *(_DWORD *)(a3 + 12) <= (unsigned int)v10 )
      {
        sub_16CD150(a3, (const void *)(a3 + 16), 0, 4, v5, v6);
        v10 = *(unsigned int *)(a3 + 8);
      }
      *(_DWORD *)(*(_QWORD *)a3 + 4 * v10) = 3;
      ++*(_DWORD *)(a3 + 8);
      return v7;
    }
    else
    {
      if ( (unsigned int)v8 >= v9 )
      {
        v12 = result;
        sub_16CD150(a3, (const void *)(a3 + 16), 0, 4, v5, v6);
        v8 = *(unsigned int *)(a3 + 8);
        result = v12;
      }
      *(_DWORD *)(*(_QWORD *)a3 + 4 * v8) = 0;
      v11 = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
      *(_DWORD *)(a3 + 8) = v11;
      if ( *(_DWORD *)(a3 + 12) <= (unsigned int)v11 )
      {
        v13 = result;
        sub_16CD150(a3, (const void *)(a3 + 16), 0, 4, v5, v6);
        v11 = *(unsigned int *)(a3 + 8);
        result = v13;
      }
      *(_DWORD *)(*(_QWORD *)a3 + 4 * v11) = 2;
      ++*(_DWORD *)(a3 + 8);
    }
  }
  return result;
}
