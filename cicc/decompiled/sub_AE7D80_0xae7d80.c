// Function: sub_AE7D80
// Address: 0xae7d80
//
__int64 __fastcall sub_AE7D80(__int64 a1, __int64 a2)
{
  _QWORD *v2; // rax
  __int64 v3; // rcx
  _QWORD *v4; // rdx
  __int64 result; // rax
  unsigned int v6; // edx
  __int64 v7; // rax

  if ( !*(_BYTE *)(a1 + 428) )
    goto LABEL_9;
  v2 = *(_QWORD **)(a1 + 408);
  v3 = *(unsigned int *)(a1 + 420);
  v4 = &v2[v3];
  if ( v2 == v4 )
  {
LABEL_8:
    if ( (unsigned int)v3 < *(_DWORD *)(a1 + 416) )
    {
      *(_DWORD *)(a1 + 420) = v3 + 1;
      *v4 = a2;
      ++*(_QWORD *)(a1 + 400);
LABEL_10:
      v7 = *(unsigned int *)(a1 + 168);
      if ( v7 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 172) )
      {
        sub_C8D5F0(a1 + 160, a1 + 176, v7 + 1, 8);
        v7 = *(unsigned int *)(a1 + 168);
      }
      *(_QWORD *)(*(_QWORD *)(a1 + 160) + 8 * v7) = a2;
      ++*(_DWORD *)(a1 + 168);
      return 1;
    }
LABEL_9:
    sub_C8CC70(a1 + 400, a2);
    result = v6;
    if ( !(_BYTE)v6 )
      return result;
    goto LABEL_10;
  }
  while ( a2 != *v2 )
  {
    if ( v4 == ++v2 )
      goto LABEL_8;
  }
  return 0;
}
