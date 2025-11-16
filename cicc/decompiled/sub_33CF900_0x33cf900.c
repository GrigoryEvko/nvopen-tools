// Function: sub_33CF900
// Address: 0x33cf900
//
__int64 __fastcall sub_33CF900(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5)
{
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rsi
  signed __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // rdx

  v5 = *(_QWORD *)(a2 + 40);
  v6 = 40LL * *(unsigned int *)(a2 + 64);
  v7 = v5 + v6;
  v8 = 0xCCCCCCCCCCCCCCCDLL * (v6 >> 3);
  v9 = v8 >> 2;
  if ( !(v8 >> 2) )
  {
LABEL_9:
    if ( v8 == 2 )
    {
      v12 = *a1;
    }
    else
    {
      if ( v8 != 3 )
      {
        a5 = 0;
        if ( v8 != 1 )
          return a5;
        v12 = *a1;
        goto LABEL_13;
      }
      v12 = *a1;
      if ( *(_QWORD *)v5 == *a1 && *(_DWORD *)(v5 + 8) == *((_DWORD *)a1 + 2) )
      {
LABEL_15:
        LOBYTE(a5) = v7 != v5;
        return a5;
      }
      v5 += 40;
    }
    if ( *(_QWORD *)v5 != v12 || *(_DWORD *)(v5 + 8) != *((_DWORD *)a1 + 2) )
    {
      v5 += 40;
LABEL_13:
      a5 = 0;
      if ( *(_QWORD *)v5 != v12 || *(_DWORD *)(v5 + 8) != *((_DWORD *)a1 + 2) )
        return a5;
      goto LABEL_15;
    }
    goto LABEL_15;
  }
  v10 = *a1;
  v11 = v5 + 160 * v9;
  while ( 1 )
  {
    if ( *(_QWORD *)v5 == v10 )
    {
      if ( *(_DWORD *)(v5 + 8) == *((_DWORD *)a1 + 2) )
        goto LABEL_15;
      if ( v10 != *(_QWORD *)(v5 + 40) )
        goto LABEL_5;
    }
    else if ( v10 != *(_QWORD *)(v5 + 40) )
    {
      goto LABEL_5;
    }
    if ( *(_DWORD *)(v5 + 48) == *((_DWORD *)a1 + 2) )
    {
      LOBYTE(a5) = v7 != v5 + 40;
      return a5;
    }
LABEL_5:
    if ( v10 == *(_QWORD *)(v5 + 80) && *(_DWORD *)(v5 + 88) == *((_DWORD *)a1 + 2) )
    {
      LOBYTE(a5) = v7 != v5 + 80;
      return a5;
    }
    if ( v10 == *(_QWORD *)(v5 + 120) && *(_DWORD *)(v5 + 128) == *((_DWORD *)a1 + 2) )
      break;
    v5 += 160;
    if ( v5 == v11 )
    {
      v8 = 0xCCCCCCCCCCCCCCCDLL * ((v7 - v5) >> 3);
      goto LABEL_9;
    }
  }
  LOBYTE(a5) = v7 != v5 + 120;
  return a5;
}
