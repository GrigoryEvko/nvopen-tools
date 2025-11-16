// Function: sub_33CA780
// Address: 0x33ca780
//
__int64 __fastcall sub_33CA780(__int64 a1)
{
  __int64 v1; // rdx
  unsigned int v2; // r8d
  _QWORD *v4; // rax
  __int64 v5; // rdx
  _QWORD *v6; // rsi
  signed __int64 v7; // rdx
  _QWORD *v8; // rcx

  v1 = *(unsigned int *)(a1 + 64);
  v2 = 0;
  if ( !(_DWORD)v1 )
    return v2;
  v4 = *(_QWORD **)(a1 + 40);
  v5 = 5 * v1;
  v6 = &v4[v5];
  v7 = 0xCCCCCCCCCCCCCCCDLL * ((v5 * 8) >> 3);
  if ( v7 >> 2 )
  {
    v8 = &v4[20 * (v7 >> 2)];
    while ( *(_DWORD *)(*v4 + 24LL) == 51 )
    {
      if ( *(_DWORD *)(v4[5] + 24LL) != 51 )
      {
        LOBYTE(v2) = v6 == v4 + 5;
        return v2;
      }
      if ( *(_DWORD *)(v4[10] + 24LL) != 51 )
      {
        LOBYTE(v2) = v6 == v4 + 10;
        return v2;
      }
      if ( *(_DWORD *)(v4[15] + 24LL) != 51 )
      {
        LOBYTE(v2) = v6 == v4 + 15;
        return v2;
      }
      v4 += 20;
      if ( v4 == v8 )
      {
        v7 = 0xCCCCCCCCCCCCCCCDLL * (v6 - v4);
        goto LABEL_12;
      }
    }
    goto LABEL_10;
  }
LABEL_12:
  if ( v7 == 2 )
  {
LABEL_22:
    if ( *(_DWORD *)(*v4 + 24LL) != 51 )
      goto LABEL_10;
    v4 += 5;
    goto LABEL_15;
  }
  if ( v7 == 3 )
  {
    if ( *(_DWORD *)(*v4 + 24LL) != 51 )
      goto LABEL_10;
    v4 += 5;
    goto LABEL_22;
  }
  v2 = 1;
  if ( v7 != 1 )
    return v2;
LABEL_15:
  v2 = 1;
  if ( *(_DWORD *)(*v4 + 24LL) == 51 )
    return v2;
LABEL_10:
  LOBYTE(v2) = v6 == v4;
  return v2;
}
