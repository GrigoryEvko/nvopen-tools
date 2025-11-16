// Function: sub_5DFB80
// Address: 0x5dfb80
//
__int64 __fastcall sub_5DFB80(_QWORD *a1, int a2, unsigned int a3, unsigned int a4)
{
  __int64 v6; // r15
  __int64 v7; // rbx
  __int64 v8; // rdx
  unsigned int v9; // eax
  __int64 result; // rax
  __int64 i; // rbx

  v6 = a1[14];
  v7 = a1[19];
  if ( v6 )
  {
    while ( (*(_BYTE *)(v6 + 156) & 3) == 3 )
    {
      v8 = *(_QWORD *)(v6 + 120);
      if ( *(_BYTE *)(v8 + 140) != 8 )
        break;
      if ( *(_BYTE *)(*(_QWORD *)(v8 + 160) + 140LL) != 14 )
      {
        if ( a2 )
          goto LABEL_15;
        goto LABEL_4;
      }
LABEL_10:
      v6 = *(_QWORD *)(v6 + 112);
      if ( !v6 )
        goto LABEL_21;
    }
    if ( a2 )
    {
LABEL_15:
      if ( v7 )
      {
        while ( 1 )
        {
          v9 = *(_DWORD *)(v6 + 64);
          if ( *(_DWORD *)(v7 + 64) >= v9 && (*(_DWORD *)(v7 + 64) != v9 || *(_WORD *)(v7 + 68) > *(_WORD *)(v6 + 68)) )
            break;
          sub_5DF1B0(v7);
          v7 = *(_QWORD *)(v7 + 112);
          if ( !v7 )
            goto LABEL_20;
        }
      }
      else
      {
LABEL_20:
        v7 = 0;
      }
    }
LABEL_4:
    if ( (*(_BYTE *)(v6 + 170) & 0x60) == 0
      && *(_BYTE *)(v6 + 177) != 5
      && (*(_BYTE *)(v6 + 173) & 2) == 0
      && (a3 || *(_BYTE *)(v6 + 136) != 1) )
    {
      sub_5D9330(v6, a3, a4);
    }
    goto LABEL_10;
  }
LABEL_21:
  if ( a2 )
  {
    while ( v7 )
    {
      sub_5DF1B0(v7);
      v7 = *(_QWORD *)(v7 + 112);
    }
  }
  result = (__int64)a1;
  for ( i = a1[15]; i; i = *(_QWORD *)(i + 112) )
  {
    while ( (*(_BYTE *)(i + 170) & 0x60) != 0 || *(_BYTE *)(i + 177) == 5 || (*(_BYTE *)(i + 173) & 2) != 0 )
    {
      i = *(_QWORD *)(i + 112);
      if ( !i )
        return result;
    }
    result = sub_5D9330(i, a3, a4);
  }
  return result;
}
