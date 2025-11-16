// Function: sub_AE7F60
// Address: 0xae7f60
//
__int64 __fastcall sub_AE7F60(__int64 a1, __int64 a2)
{
  int v2; // eax
  _QWORD *v4; // rax
  __int64 v5; // rcx
  _QWORD *v6; // rdx
  char v7; // dl
  __int64 v8; // rax

  if ( !a2 )
    return 0;
  if ( (*(_BYTE *)(a2 - 16) & 2) != 0 )
    v2 = *(_DWORD *)(a2 - 24);
  else
    v2 = (*(_WORD *)(a2 - 16) >> 6) & 0xF;
  if ( !v2 )
    return 0;
  if ( !*(_BYTE *)(a1 + 428) )
    goto LABEL_13;
  v4 = *(_QWORD **)(a1 + 408);
  v5 = *(unsigned int *)(a1 + 420);
  v6 = &v4[v5];
  if ( v4 != v6 )
  {
    while ( a2 != *v4 )
    {
      if ( v6 == ++v4 )
        goto LABEL_18;
    }
    return 0;
  }
LABEL_18:
  if ( (unsigned int)v5 < *(_DWORD *)(a1 + 416) )
  {
    *(_DWORD *)(a1 + 420) = v5 + 1;
    *v6 = a2;
    ++*(_QWORD *)(a1 + 400);
  }
  else
  {
LABEL_13:
    sub_C8CC70(a1 + 400, a2);
    if ( !v7 )
      return 0;
  }
  v8 = *(unsigned int *)(a1 + 328);
  if ( v8 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 332) )
  {
    sub_C8D5F0(a1 + 320, a1 + 336, v8 + 1, 8);
    v8 = *(unsigned int *)(a1 + 328);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 320) + 8 * v8) = a2;
  ++*(_DWORD *)(a1 + 328);
  return 1;
}
