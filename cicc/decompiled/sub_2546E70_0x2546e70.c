// Function: sub_2546E70
// Address: 0x2546e70
//
__int64 __fastcall sub_2546E70(__int64 a1, __int64 a2, const void **a3)
{
  __int64 v4; // r14
  __int64 v5; // rax
  __int64 v6; // rbx
  __int64 v7; // r14
  __int64 v8; // r12

  v4 = (a2 - a1) >> 6;
  v5 = (a2 - a1) >> 4;
  v6 = a1;
  if ( v4 > 0 )
  {
    v7 = a1 + (v4 << 6);
    while ( 1 )
    {
      if ( *(_DWORD *)(v6 + 8) > 0x40u )
      {
        if ( sub_C43C50(v6, a3) )
          return v6;
      }
      else if ( *(const void **)v6 == *a3 )
      {
        return v6;
      }
      v8 = v6 + 16;
      if ( *(_DWORD *)(v6 + 24) <= 0x40u )
      {
        if ( *(const void **)(v6 + 16) == *a3 )
          return v8;
        v8 = v6 + 32;
        if ( *(_DWORD *)(v6 + 40) <= 0x40u )
          goto LABEL_17;
LABEL_7:
        if ( sub_C43C50(v8, a3) )
          return v8;
        v8 = v6 + 48;
        if ( *(_DWORD *)(v6 + 56) <= 0x40u )
          goto LABEL_19;
LABEL_9:
        if ( sub_C43C50(v8, a3) )
          return v8;
        v6 += 64;
        if ( v7 == v6 )
          goto LABEL_21;
      }
      else
      {
        if ( sub_C43C50(v6 + 16, a3) )
          return v8;
        v8 = v6 + 32;
        if ( *(_DWORD *)(v6 + 40) > 0x40u )
          goto LABEL_7;
LABEL_17:
        if ( *(const void **)(v6 + 32) == *a3 )
          return v8;
        v8 = v6 + 48;
        if ( *(_DWORD *)(v6 + 56) > 0x40u )
          goto LABEL_9;
LABEL_19:
        if ( *(const void **)(v6 + 48) == *a3 )
          return v8;
        v6 += 64;
        if ( v7 == v6 )
        {
LABEL_21:
          v5 = (a2 - v6) >> 4;
          break;
        }
      }
    }
  }
  if ( v5 != 2 )
  {
    if ( v5 != 3 )
    {
      v8 = a2;
      if ( v5 != 1 )
        return v8;
      goto LABEL_25;
    }
    if ( *(_DWORD *)(v6 + 8) <= 0x40u )
    {
      v8 = v6;
      if ( *(const void **)v6 == *a3 )
        return v8;
    }
    else
    {
      v8 = v6;
      if ( sub_C43C50(v6, a3) )
        return v8;
    }
    v6 += 16;
  }
  if ( *(_DWORD *)(v6 + 8) <= 0x40u )
  {
    v8 = v6;
    if ( *(const void **)v6 == *a3 )
      return v8;
  }
  else
  {
    v8 = v6;
    if ( sub_C43C50(v6, a3) )
      return v8;
  }
  v6 += 16;
LABEL_25:
  if ( *(_DWORD *)(v6 + 8) <= 0x40u )
  {
    if ( *(const void **)v6 != *a3 )
      return a2;
    return v6;
  }
  else
  {
    if ( !sub_C43C50(v6, a3) )
      return a2;
    return v6;
  }
}
