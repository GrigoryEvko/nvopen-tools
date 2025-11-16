// Function: sub_15AB650
// Address: 0x15ab650
//
__int64 __fastcall sub_15AB650(__int64 a1, __int64 a2)
{
  _QWORD *v4; // rax
  char v6; // dl
  __int64 v7; // rax
  _QWORD *v8; // rsi
  unsigned int v9; // edi
  _QWORD *v10; // rcx

  if ( !a2 )
    return 0;
  if ( !*(_DWORD *)(a2 + 8) )
    return 0;
  v4 = *(_QWORD **)(a1 + 408);
  if ( *(_QWORD **)(a1 + 416) != v4 )
    goto LABEL_5;
  v8 = &v4[*(unsigned int *)(a1 + 428)];
  v9 = *(_DWORD *)(a1 + 428);
  if ( v4 != v8 )
  {
    v10 = 0;
    while ( *v4 != a2 )
    {
      if ( *v4 == -2 )
        v10 = v4;
      if ( v8 == ++v4 )
      {
        if ( !v10 )
          goto LABEL_19;
        *v10 = a2;
        --*(_DWORD *)(a1 + 432);
        ++*(_QWORD *)(a1 + 400);
        goto LABEL_6;
      }
    }
    return 0;
  }
LABEL_19:
  if ( v9 < *(_DWORD *)(a1 + 424) )
  {
    *(_DWORD *)(a1 + 428) = v9 + 1;
    *v8 = a2;
    ++*(_QWORD *)(a1 + 400);
  }
  else
  {
LABEL_5:
    sub_16CCBA0(a1 + 400, a2);
    if ( !v6 )
      return 0;
  }
LABEL_6:
  v7 = *(unsigned int *)(a1 + 328);
  if ( (unsigned int)v7 >= *(_DWORD *)(a1 + 332) )
  {
    sub_16CD150(a1 + 320, a1 + 336, 0, 8);
    v7 = *(unsigned int *)(a1 + 328);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 320) + 8 * v7) = a2;
  ++*(_DWORD *)(a1 + 328);
  return 1;
}
