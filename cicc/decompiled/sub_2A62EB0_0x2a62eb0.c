// Function: sub_2A62EB0
// Address: 0x2a62eb0
//
__int64 __fastcall sub_2A62EB0(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v6; // rax
  __int64 result; // rax
  unsigned int v8; // edx
  __int64 v9; // rax

  if ( !*(_BYTE *)(a1 + 68) )
    goto LABEL_9;
  v6 = *(__int64 **)(a1 + 48);
  a4 = *(unsigned int *)(a1 + 60);
  a3 = &v6[a4];
  if ( v6 == a3 )
  {
LABEL_8:
    if ( (unsigned int)a4 < *(_DWORD *)(a1 + 56) )
    {
      *(_DWORD *)(a1 + 60) = a4 + 1;
      *a3 = a2;
      ++*(_QWORD *)(a1 + 40);
LABEL_10:
      v9 = *(unsigned int *)(a1 + 1984);
      if ( v9 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 1988) )
      {
        sub_C8D5F0(a1 + 1976, (const void *)(a1 + 1992), v9 + 1, 8u, a5, a6);
        v9 = *(unsigned int *)(a1 + 1984);
      }
      *(_QWORD *)(*(_QWORD *)(a1 + 1976) + 8 * v9) = a2;
      ++*(_DWORD *)(a1 + 1984);
      return 1;
    }
LABEL_9:
    sub_C8CC70(a1 + 40, a2, (__int64)a3, a4, a5, a6);
    result = v8;
    if ( !(_BYTE)v8 )
      return result;
    goto LABEL_10;
  }
  while ( a2 != *v6 )
  {
    if ( a3 == ++v6 )
      goto LABEL_8;
  }
  return 0;
}
