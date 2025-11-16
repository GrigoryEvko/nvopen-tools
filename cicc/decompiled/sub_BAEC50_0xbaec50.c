// Function: sub_BAEC50
// Address: 0xbaec50
//
__int64 __fastcall sub_BAEC50(_QWORD *a1, char a2)
{
  unsigned __int64 v3; // rdx
  char *v4; // rax
  char *v5; // rsi
  unsigned int v6; // r8d
  __int64 v8; // rcx
  __int64 v9; // rdx
  char *v10; // rdx

  v3 = *a1 & 0xFFFFFFFFFFFFFFF8LL;
  v4 = *(char **)(v3 + 24);
  v5 = *(char **)(v3 + 32);
  if ( !a2 )
  {
    v6 = 0;
    v8 = (v5 - v4) >> 3;
    if ( v4 == v5 )
      return v6;
    v9 = (v5 - v4) >> 5;
    if ( v9 > 0 )
    {
      v10 = &v4[32 * v9];
      while ( (*(_BYTE *)(*(_QWORD *)v4 + 13LL) & 1) != 0 )
      {
        if ( (*(_BYTE *)(*((_QWORD *)v4 + 1) + 13LL) & 1) == 0 )
        {
          LOBYTE(v6) = v5 == v4 + 8;
          return v6;
        }
        if ( (*(_BYTE *)(*((_QWORD *)v4 + 2) + 13LL) & 1) == 0 )
        {
          LOBYTE(v6) = v5 == v4 + 16;
          return v6;
        }
        if ( (*(_BYTE *)(*((_QWORD *)v4 + 3) + 13LL) & 1) == 0 )
        {
          LOBYTE(v6) = v5 == v4 + 24;
          return v6;
        }
        v4 += 32;
        if ( v10 == v4 )
        {
          v8 = (v5 - v4) >> 3;
          goto LABEL_15;
        }
      }
      goto LABEL_13;
    }
LABEL_15:
    if ( v8 != 2 )
    {
      if ( v8 != 3 )
      {
        v6 = 1;
        if ( v8 != 1 )
          return v6;
        goto LABEL_18;
      }
      if ( (*(_BYTE *)(*(_QWORD *)v4 + 13LL) & 1) == 0 )
      {
LABEL_13:
        LOBYTE(v6) = v4 == v5;
        return v6;
      }
      v4 += 8;
    }
    if ( (*(_BYTE *)(*(_QWORD *)v4 + 13LL) & 1) != 0 )
    {
      v4 += 8;
LABEL_18:
      v6 = *(_BYTE *)(*(_QWORD *)v4 + 13LL) & 1;
      if ( (*(_BYTE *)(*(_QWORD *)v4 + 13LL) & 1) != 0 )
        return v6;
      goto LABEL_13;
    }
    goto LABEL_13;
  }
  v6 = 0;
  if ( v4 != v5 )
    return *(_BYTE *)(*(_QWORD *)v4 + 13LL) & 1;
  return v6;
}
