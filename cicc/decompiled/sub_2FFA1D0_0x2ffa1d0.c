// Function: sub_2FFA1D0
// Address: 0x2ffa1d0
//
char __fastcall sub_2FFA1D0(_QWORD *a1, unsigned __int64 a2, unsigned int a3, char a4)
{
  __int64 v4; // r12
  __int64 v6; // rax
  char result; // al
  __int64 v8; // rdx
  __int64 v9; // rdx
  __int64 v10; // rax
  __int16 v11; // ax

  v4 = a3;
  while ( (unsigned int)(v4 - 1) > 0x3FFFFFFE )
  {
    result = sub_2FF9BD0(a1, a2, v4);
    if ( !result )
      return result;
    v8 = a1[4];
    if ( (int)v4 < 0 )
      v9 = *(_QWORD *)(*(_QWORD *)(v8 + 56) + 16 * (v4 & 0x7FFFFFFF) + 8);
    else
      v9 = *(_QWORD *)(*(_QWORD *)(v8 + 304) + 8 * v4);
    if ( v9 )
    {
      if ( (*(_BYTE *)(v9 + 3) & 0x10) == 0 )
      {
        v9 = *(_QWORD *)(v9 + 32);
        if ( v9 )
        {
          if ( (*(_BYTE *)(v9 + 3) & 0x10) == 0 )
            BUG();
        }
      }
    }
    v10 = *(_QWORD *)(v9 + 32);
    if ( v10 && (*(_BYTE *)(v10 + 3) & 0x10) != 0 )
      return 1;
    a2 = *(_QWORD *)(v9 + 16);
    v11 = *(_WORD *)(a2 + 68);
    if ( v11 == 20 )
    {
      v4 = *(unsigned int *)(*(_QWORD *)(a2 + 32) + 48LL);
    }
    else
    {
      if ( v11 != 9 && v11 != 12 )
        return 1;
      v4 = *(unsigned int *)(*(_QWORD *)(a2 + 32) + 88LL);
    }
  }
  if ( a4 )
    return 1;
  v6 = *(_QWORD *)(*(_QWORD *)(a1[4] + 304LL) + 8LL * (unsigned int)v4);
  if ( v6 )
  {
    if ( (*(_BYTE *)(v6 + 3) & 0x10) == 0 )
    {
      v6 = *(_QWORD *)(v6 + 32);
      if ( v6 )
      {
        while ( (*(_BYTE *)(v6 + 3) & 0x10) != 0 )
        {
LABEL_8:
          v6 = *(_QWORD *)(v6 + 32);
          if ( !v6 )
            return 1;
        }
        return sub_2FF9BD0(a1, a2, v4);
      }
      return 1;
    }
    while ( 1 )
    {
      v6 = *(_QWORD *)(v6 + 32);
      if ( !v6 )
        break;
      if ( (*(_BYTE *)(v6 + 3) & 0x10) == 0 )
        goto LABEL_8;
    }
  }
  return sub_2FF9BD0(a1, a2, v4);
}
