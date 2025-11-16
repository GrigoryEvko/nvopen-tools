// Function: sub_AE7B90
// Address: 0xae7b90
//
__int64 __fastcall sub_AE7B90(__int64 a1, __int64 a2)
{
  _QWORD *v2; // rax
  __int64 v3; // rcx
  _QWORD *v4; // rdx
  char v6; // dl
  __int64 v7; // rax

  if ( !a2 )
    return 0;
  if ( *(_BYTE *)(a1 + 428) )
  {
    v2 = *(_QWORD **)(a1 + 408);
    v3 = *(unsigned int *)(a1 + 420);
    v4 = &v2[v3];
    if ( v2 != v4 )
    {
      while ( a2 != *v2 )
      {
        if ( v4 == ++v2 )
          goto LABEL_8;
      }
      return 0;
    }
LABEL_8:
    if ( (unsigned int)v3 < *(_DWORD *)(a1 + 416) )
    {
      *(_DWORD *)(a1 + 420) = v3 + 1;
      *v4 = a2;
      ++*(_QWORD *)(a1 + 400);
      goto LABEL_10;
    }
  }
  sub_C8CC70(a1 + 400, a2);
  if ( !v6 )
    return 0;
LABEL_10:
  v7 = *(unsigned int *)(a1 + 248);
  if ( v7 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 252) )
  {
    sub_C8D5F0(a1 + 240, a1 + 256, v7 + 1, 8);
    v7 = *(unsigned int *)(a1 + 248);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 240) + 8 * v7) = a2;
  ++*(_DWORD *)(a1 + 248);
  return 1;
}
