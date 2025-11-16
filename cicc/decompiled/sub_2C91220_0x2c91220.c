// Function: sub_2C91220
// Address: 0x2c91220
//
bool __fastcall sub_2C91220(__int64 a1)
{
  __int16 v2; // ax
  __int64 v3; // rdi
  unsigned int v4; // esi
  __int64 v5; // rax
  __int64 v7; // r12
  __int64 v8; // r13
  unsigned __int8 *v9; // rax
  __int64 v10; // rdx
  int v11; // edx
  bool v12; // al

  while ( 1 )
  {
    v2 = *(_WORD *)(a1 + 24);
    if ( (unsigned __int16)(v2 - 8) <= 5u || (unsigned __int16)(v2 - 5) <= 1u )
      break;
    while ( 1 )
    {
      if ( !v2 )
      {
        v3 = *(_QWORD *)(a1 + 32);
        v4 = *(_DWORD *)(v3 + 32);
        v5 = *(_QWORD *)(v3 + 24);
        if ( v4 > 0x40 )
          v5 = *(_QWORD *)(v5 + 8LL * ((v4 - 1) >> 6));
        return (v5 & (1LL << ((unsigned __int8)v4 - 1))) != 0;
      }
      if ( (unsigned __int16)(v2 - 2) > 2u && v2 != 14 )
        break;
      a1 = *(_QWORD *)(a1 + 32);
      v2 = *(_WORD *)(a1 + 24);
      if ( (unsigned __int16)(v2 - 5) <= 1u || (unsigned __int16)(v2 - 8) <= 5u )
        goto LABEL_11;
    }
    if ( v2 != 7 )
    {
      if ( v2 == 15 )
      {
        v9 = sub_CEFC00(*(unsigned __int8 **)(a1 - 8), (unsigned __int8 **)1);
        if ( *v9 <= 0x1Cu )
          return 0;
        if ( *v9 == 85 )
        {
          v10 = *((_QWORD *)v9 - 4);
          if ( v10 )
          {
            if ( !*(_BYTE *)v10 && *(_QWORD *)(v10 + 24) == *((_QWORD *)v9 + 10) && (*(_BYTE *)(v10 + 33) & 0x20) != 0 )
            {
              v11 = *(_DWORD *)(v10 + 36);
              if ( (unsigned int)(v11 - 9370) <= 2 )
                return 0;
              v12 = 1;
              if ( (unsigned int)(v11 - 9307) <= 0x37 )
                v12 = ((0xE7000000000007uLL >> ((unsigned __int8)v11 - 91)) & 1) == 0;
              if ( v11 == 9374 || !v12 )
                return 0;
            }
          }
        }
      }
      return 1;
    }
    if ( (unsigned __int8)sub_2C91220(*(_QWORD *)(a1 + 32)) )
      return 1;
    a1 = *(_QWORD *)(a1 + 40);
  }
LABEL_11:
  v7 = 0;
  v8 = 8LL * (unsigned int)*(_QWORD *)(a1 + 40);
  if ( !(unsigned int)*(_QWORD *)(a1 + 40) )
    return 0;
  while ( !(unsigned __int8)sub_2C91220(*(_QWORD *)(*(_QWORD *)(a1 + 32) + v7)) )
  {
    v7 += 8;
    if ( v8 == v7 )
      return 0;
  }
  return 1;
}
