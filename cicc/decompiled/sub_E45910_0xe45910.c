// Function: sub_E45910
// Address: 0xe45910
//
__int64 __fastcall sub_E45910(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rdx
  __int64 *v3; // rbx
  __int64 *v4; // r13
  __int64 v5; // rax
  __int64 *v6; // r12
  unsigned int v7; // r8d

  v1 = 8LL * *(unsigned int *)(a1 + 12);
  if ( v1 && (*(_DWORD *)(a1 + 8) & 0x400) != 0 && ((*(_DWORD *)(a1 + 8) >> 8) & 2) == 0 )
  {
    v2 = v1 >> 3;
    v3 = *(__int64 **)(a1 + 16);
    v4 = &v3[(unsigned __int64)v1 / 8];
    v5 = v1 >> 5;
    if ( v5 )
    {
      v6 = &v3[4 * v5];
      while ( (unsigned __int8)sub_BCBCB0(*v3) )
      {
        if ( !(unsigned __int8)sub_BCBCB0(v3[1]) )
        {
          LOBYTE(v7) = v4 == v3 + 1;
          return v7;
        }
        if ( !(unsigned __int8)sub_BCBCB0(v3[2]) )
        {
          LOBYTE(v7) = v4 == v3 + 2;
          return v7;
        }
        if ( !(unsigned __int8)sub_BCBCB0(v3[3]) )
        {
          LOBYTE(v7) = v4 == v3 + 3;
          return v7;
        }
        v3 += 4;
        if ( v3 == v6 )
        {
          v2 = v4 - v3;
          goto LABEL_15;
        }
      }
      goto LABEL_11;
    }
LABEL_15:
    if ( v2 != 2 )
    {
      if ( v2 != 3 )
      {
        v7 = 1;
        if ( v2 != 1 )
          return v7;
        goto LABEL_18;
      }
      if ( !(unsigned __int8)sub_BCBCB0(*v3) )
      {
LABEL_11:
        LOBYTE(v7) = v4 == v3;
        return v7;
      }
      ++v3;
    }
    if ( (unsigned __int8)sub_BCBCB0(*v3) )
    {
      ++v3;
LABEL_18:
      v7 = sub_BCBCB0(*v3);
      if ( (_BYTE)v7 )
        return v7;
      goto LABEL_11;
    }
    goto LABEL_11;
  }
  return 0;
}
