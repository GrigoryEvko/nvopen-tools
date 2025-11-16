// Function: sub_3074410
// Address: 0x3074410
//
__int64 __fastcall sub_3074410(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v6; // rdx
  __int64 v7; // r9
  __int64 result; // rax
  __int64 v9; // rax
  __int64 v10; // rdx
  unsigned __int8 v11; // [rsp+Fh] [rbp-21h]
  int v12; // [rsp+18h] [rbp-18h]

  if ( a3 > 0x22E0 )
  {
    if ( a3 - 8930 > 2 )
    {
LABEL_3:
      v6 = sub_3071AE0(a3);
      result = HIDWORD(v6);
      v12 = v6;
      if ( BYTE4(v6) )
      {
        v10 = *(unsigned int *)(a2 + 8);
        if ( v10 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
        {
          v11 = result;
          sub_C8D5F0(a2, (const void *)(a2 + 16), v10 + 1, 4u, v10 + 1, v7);
          v10 = *(unsigned int *)(a2 + 8);
          result = v11;
        }
        *(_DWORD *)(*(_QWORD *)a2 + 4 * v10) = v12;
        ++*(_DWORD *)(a2 + 8);
      }
      return result;
    }
  }
  else if ( a3 <= 0x22DE )
  {
    goto LABEL_3;
  }
  v9 = *(unsigned int *)(a2 + 8);
  if ( v9 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
  {
    sub_C8D5F0(a2, (const void *)(a2 + 16), v9 + 1, 4u, a5, a6);
    v9 = *(unsigned int *)(a2 + 8);
  }
  *(_DWORD *)(*(_QWORD *)a2 + 4 * v9) = 0;
  ++*(_DWORD *)(a2 + 8);
  return 1;
}
