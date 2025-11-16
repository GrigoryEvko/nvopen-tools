// Function: sub_E456C0
// Address: 0xe456c0
//
__int64 __fastcall sub_E456C0(__int64 a1)
{
  __int64 v1; // r8
  __int64 v2; // rdx
  __int64 *v3; // rax
  __int64 v4; // r10
  int v5; // esi
  int v6; // edi
  __int64 *v7; // rcx
  bool v8; // r9
  __int64 v10; // rsi
  int v11; // r10d
  __int64 v12; // rsi
  __int64 *v13; // r11
  int v14; // r10d
  __int64 v15; // rsi
  int v16; // r10d
  char *v17; // rdx

  LODWORD(v1) = 0;
  if ( (*(_DWORD *)(a1 + 8) & 0x400) == 0 )
    return (unsigned int)v1;
  if ( ((*(_DWORD *)(a1 + 8) >> 8) & 2) != 0 )
    return (unsigned int)v1;
  v2 = *(unsigned int *)(a1 + 12);
  if ( !(v2 * 8) )
    return (unsigned int)v1;
  v3 = *(__int64 **)(a1 + 16);
  v4 = *v3;
  v5 = *(unsigned __int8 *)(*v3 + 8);
  if ( (unsigned int)(v5 - 17) > 1 )
    return (unsigned int)v1;
  v6 = *(_DWORD *)(v4 + 32);
  v7 = &v3[v2];
  v8 = (_BYTE)v5 == 18;
  v1 = (v2 * 8) >> 5;
  if ( (v2 * 8) >> 5 )
  {
    v1 = (__int64)&v3[4 * v1];
    while ( v8 == ((_BYTE)v5 == 18) )
    {
      v10 = v3[1];
      v11 = *(unsigned __int8 *)(v10 + 8);
      if ( (unsigned int)(v11 - 17) > 1 || v6 != *(_DWORD *)(v10 + 32) || v8 != ((_BYTE)v11 == 18) )
      {
        LOBYTE(v1) = v3 + 1 == v7;
        return (unsigned int)v1;
      }
      v12 = v3[2];
      v13 = v3 + 2;
      v14 = *(unsigned __int8 *)(v12 + 8);
      if ( (unsigned int)(v14 - 17) > 1
        || v6 != *(_DWORD *)(v12 + 32)
        || v8 != ((_BYTE)v14 == 18)
        || (v15 = v3[3], v13 = v3 + 3, v16 = *(unsigned __int8 *)(v15 + 8), (unsigned int)(v16 - 17) > 1)
        || v6 != *(_DWORD *)(v15 + 32)
        || v8 != ((_BYTE)v16 == 18) )
      {
        LOBYTE(v1) = v7 == v13;
        return (unsigned int)v1;
      }
      v3 += 4;
      if ( (__int64 *)v1 == v3 )
      {
        v17 = (char *)v7 - v1;
        if ( (__int64 *)((char *)v7 - v1) == (__int64 *)16 )
          goto LABEL_32;
        if ( v17 != (char *)24 )
        {
          if ( v17 == (char *)8 )
            goto LABEL_36;
          goto LABEL_28;
        }
        v4 = *(_QWORD *)v1;
        v5 = *(unsigned __int8 *)(*(_QWORD *)v1 + 8LL);
        if ( (unsigned int)(v5 - 17) <= 1 )
          goto LABEL_29;
        LOBYTE(v1) = v7 == (__int64 *)v1;
        return (unsigned int)v1;
      }
      v5 = *(unsigned __int8 *)(*v3 + 8);
      if ( (unsigned int)(v5 - 17) > 1 || v6 != *(_DWORD *)(*v3 + 32) )
        goto LABEL_8;
    }
    goto LABEL_8;
  }
  if ( v2 == 2 )
    goto LABEL_37;
  if ( v2 != 3 )
  {
    if ( v2 != 1 )
    {
LABEL_28:
      LODWORD(v1) = 1;
      return (unsigned int)v1;
    }
    goto LABEL_38;
  }
LABEL_29:
  if ( v6 == *(_DWORD *)(v4 + 32) && v8 == ((_BYTE)v5 == 18) )
  {
    ++v3;
LABEL_32:
    v4 = *v3;
    v5 = *(unsigned __int8 *)(*v3 + 8);
    if ( (unsigned int)(v5 - 17) <= 1 )
    {
LABEL_37:
      if ( v6 == *(_DWORD *)(v4 + 32) && v8 == ((_BYTE)v5 == 18) )
      {
        ++v3;
LABEL_36:
        v1 = *v3;
        v5 = *(unsigned __int8 *)(*v3 + 8);
        if ( (unsigned int)(v5 - 17) <= 1 && v6 == *(_DWORD *)(v1 + 32) )
        {
LABEL_38:
          LODWORD(v1) = 1;
          if ( v8 == ((_BYTE)v5 == 18) )
            return (unsigned int)v1;
        }
      }
    }
  }
LABEL_8:
  LOBYTE(v1) = v7 == v3;
  return (unsigned int)v1;
}
