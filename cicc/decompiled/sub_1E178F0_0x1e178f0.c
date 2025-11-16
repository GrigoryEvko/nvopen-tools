// Function: sub_1E178F0
// Address: 0x1e178f0
//
__int64 __fastcall sub_1E178F0(__int64 a1)
{
  __int64 v1; // rax
  __int16 v2; // dx
  __int64 v3; // rdx
  unsigned int v4; // r8d
  __int64 v5; // rdx
  _QWORD *v6; // rax
  _QWORD *v7; // rsi
  __int64 v8; // rcx
  __int64 v9; // rdx
  _QWORD *v10; // rdx
  __int64 v12; // rax
  __int16 v13; // dx
  char v14; // al
  __int16 v15; // ax
  __int64 v16; // rax
  unsigned int v17; // eax

  v1 = *(_QWORD *)(a1 + 16);
  if ( *(_WORD *)v1 != 1 || (*(_BYTE *)(*(_QWORD *)(a1 + 32) + 64LL) & 0x10) == 0 )
  {
    v2 = *(_WORD *)(a1 + 46);
    if ( (v2 & 4) != 0 || (v2 & 8) == 0 )
    {
      if ( (*(_QWORD *)(v1 + 8) & 0x20000LL) != 0 )
        goto LABEL_5;
    }
    else if ( sub_1E15D00(a1, 0x20000u, 1) )
    {
      goto LABEL_5;
    }
    v12 = *(_QWORD *)(a1 + 16);
    if ( *(_WORD *)v12 != 1 || (*(_BYTE *)(*(_QWORD *)(a1 + 32) + 64LL) & 8) == 0 )
    {
      v13 = *(_WORD *)(a1 + 46);
      if ( (v13 & 4) != 0 || (v13 & 8) == 0 )
        v14 = WORD1(*(_QWORD *)(v12 + 8)) & 1;
      else
        v14 = sub_1E15D00(a1, 0x10000u, 1);
      if ( !v14 )
      {
        v15 = *(_WORD *)(a1 + 46);
        if ( (v15 & 4) == 0 && (v15 & 8) != 0 )
          LOBYTE(v16) = sub_1E15D00(a1, 0x10u, 1);
        else
          v16 = (*(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) >> 4) & 1LL;
        if ( !(_BYTE)v16 )
        {
          LOBYTE(v17) = sub_1E17880(a1);
          v4 = v17;
          if ( !(_BYTE)v17 )
            return v4;
        }
      }
    }
  }
LABEL_5:
  v3 = *(unsigned __int8 *)(a1 + 49);
  v4 = 1;
  if ( (_BYTE)v3 )
  {
    v5 = 8 * v3;
    v6 = *(_QWORD **)(a1 + 56);
    v7 = &v6[(unsigned __int64)v5 / 8];
    v8 = v5 >> 3;
    v9 = v5 >> 5;
    if ( v9 )
    {
      v10 = &v6[4 * v9];
      while ( (*(_BYTE *)(*v6 + 32LL) & 4) == 0 )
      {
        if ( (*(_BYTE *)(v6[1] + 32LL) & 4) != 0 )
        {
          LOBYTE(v4) = v7 != v6 + 1;
          return v4;
        }
        if ( (*(_BYTE *)(v6[2] + 32LL) & 4) != 0 )
        {
          LOBYTE(v4) = v7 != v6 + 2;
          return v4;
        }
        if ( (*(_BYTE *)(v6[3] + 32LL) & 4) != 0 )
        {
          LOBYTE(v4) = v7 != v6 + 3;
          return v4;
        }
        v6 += 4;
        if ( v10 == v6 )
        {
          v8 = v7 - v6;
          goto LABEL_31;
        }
      }
      goto LABEL_13;
    }
LABEL_31:
    if ( v8 != 2 )
    {
      if ( v8 != 3 )
      {
        v4 = 0;
        if ( v8 != 1 )
          return v4;
        goto LABEL_34;
      }
      if ( (*(_BYTE *)(*v6 + 32LL) & 4) != 0 )
      {
LABEL_13:
        LOBYTE(v4) = v7 != v6;
        return v4;
      }
      ++v6;
    }
    if ( (*(_BYTE *)(*v6 + 32LL) & 4) == 0 )
    {
      ++v6;
LABEL_34:
      v4 = 0;
      if ( (*(_BYTE *)(*v6 + 32LL) & 4) == 0 )
        return v4;
      goto LABEL_13;
    }
    goto LABEL_13;
  }
  return v4;
}
