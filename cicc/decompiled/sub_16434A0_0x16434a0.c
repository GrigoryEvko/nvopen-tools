// Function: sub_16434A0
// Address: 0x16434a0
//
_BOOL8 __fastcall sub_16434A0(__int64 a1, __int64 a2)
{
  _BOOL4 v2; // r14d
  _QWORD *v4; // rax
  char v5; // dl
  _QWORD *v6; // rbx
  __int64 v7; // r15
  _QWORD *v8; // r14
  unsigned __int64 v9; // rax
  __int64 v11; // rdx
  _QWORD *v12; // rsi
  _QWORD *v13; // rcx

  v2 = 1;
  if ( (*(_DWORD *)(a1 + 8) & 0x800) != 0 )
    return v2;
  v2 = (*(_DWORD *)(a1 + 8) & 0x100) == 0;
  if ( (*(_DWORD *)(a1 + 8) & 0x100) == 0 )
    return 0;
  if ( !a2 )
  {
LABEL_6:
    v6 = *(_QWORD **)(a1 + 16);
    v7 = 35454;
    v8 = &v6[*(unsigned int *)(a1 + 12)];
    if ( v6 == v8 )
    {
LABEL_15:
      *(_DWORD *)(a1 + 8) |= 0x800u;
      return 1;
    }
    while ( 1 )
    {
      while ( 1 )
      {
        v9 = *(unsigned __int8 *)(*v6 + 8LL);
        if ( (unsigned __int8)v9 > 0xFu || !_bittest64(&v7, v9) )
          break;
        if ( v8 == ++v6 )
          goto LABEL_15;
      }
      if ( (unsigned int)(v9 - 13) > 1 && (_DWORD)v9 != 16 || !(unsigned __int8)sub_16435F0(*v6, a2) )
        return 0;
      if ( v8 == ++v6 )
        goto LABEL_15;
    }
  }
  v4 = *(_QWORD **)(a2 + 8);
  if ( *(_QWORD **)(a2 + 16) == v4 )
  {
    v11 = *(unsigned int *)(a2 + 28);
    v12 = &v4[v11];
    if ( v4 != v12 )
    {
      v13 = 0;
      while ( a1 != *v4 )
      {
        if ( *v4 == -2 )
          v13 = v4;
        if ( v12 == ++v4 )
        {
          if ( !v13 )
            goto LABEL_26;
          *v13 = a1;
          --*(_DWORD *)(a2 + 32);
          ++*(_QWORD *)a2;
          goto LABEL_6;
        }
      }
      return v2;
    }
LABEL_26:
    if ( (unsigned int)v11 < *(_DWORD *)(a2 + 24) )
    {
      *(_DWORD *)(a2 + 28) = v11 + 1;
      *v12 = a1;
      ++*(_QWORD *)a2;
      goto LABEL_6;
    }
  }
  sub_16CCBA0(a2, a1);
  if ( v5 )
    goto LABEL_6;
  return v2;
}
