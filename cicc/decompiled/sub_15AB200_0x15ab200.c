// Function: sub_15AB200
// Address: 0x15ab200
//
__int64 __fastcall sub_15AB200(__int64 a1, __int64 a2)
{
  _QWORD *v4; // rax
  char v5; // dl
  _QWORD *v7; // rsi
  unsigned int v8; // edi
  _QWORD *v9; // rcx
  __int64 v10; // rax

  if ( !a2 )
    return 0;
  v4 = *(_QWORD **)(a1 + 408);
  if ( *(_QWORD **)(a1 + 416) != v4 )
    goto LABEL_3;
  v7 = &v4[*(unsigned int *)(a1 + 428)];
  v8 = *(_DWORD *)(a1 + 428);
  if ( v4 != v7 )
  {
    v9 = 0;
    while ( a2 != *v4 )
    {
      if ( *v4 == -2 )
        v9 = v4;
      if ( v7 == ++v4 )
      {
        if ( !v9 )
          goto LABEL_18;
        *v9 = a2;
        --*(_DWORD *)(a1 + 432);
        ++*(_QWORD *)(a1 + 400);
        goto LABEL_14;
      }
    }
    return 0;
  }
LABEL_18:
  if ( v8 < *(_DWORD *)(a1 + 424) )
  {
    *(_DWORD *)(a1 + 428) = v8 + 1;
    *v7 = a2;
    ++*(_QWORD *)(a1 + 400);
  }
  else
  {
LABEL_3:
    sub_16CCBA0(a1 + 400, a2);
    if ( !v5 )
      return 0;
  }
LABEL_14:
  v10 = *(unsigned int *)(a1 + 248);
  if ( (unsigned int)v10 >= *(_DWORD *)(a1 + 252) )
  {
    sub_16CD150(a1 + 240, a1 + 256, 0, 8);
    v10 = *(unsigned int *)(a1 + 248);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 240) + 8 * v10) = a2;
  ++*(_DWORD *)(a1 + 248);
  return 1;
}
