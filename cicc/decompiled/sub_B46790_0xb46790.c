// Function: sub_B46790
// Address: 0xb46790
//
__int64 __fastcall sub_B46790(unsigned __int8 *a1, unsigned int a2)
{
  unsigned int v3; // r8d
  __int64 v5; // rax
  __int64 v6; // rsi
  unsigned int v7; // eax
  __int64 v8; // r9
  __int64 v9; // rax
  __int64 v10; // r9
  __int64 v11; // rdx
  __int64 v12; // rcx
  char v13; // al

  if ( *a1 == 39 )
    return !(*((_WORD *)a1 + 1) & 1);
  if ( (unsigned int)*a1 - 29 > 0xA )
  {
    v3 = a2;
    if ( *a1 == 80 )
      return v3;
    if ( *a1 == 85 )
    {
      v13 = sub_A73ED0((_QWORD *)a1 + 9, 41);
      v3 = 0;
      if ( !v13 )
        return (unsigned int)sub_B49560(a1, 41) ^ 1;
      return v3;
    }
    return 0;
  }
  if ( *a1 != 35 )
  {
    if ( *a1 == 37 )
      return !(*((_WORD *)a1 + 1) & 1);
    if ( *a1 == 34 )
    {
      v5 = sub_AA4FF0(*((_QWORD *)a1 - 8));
      v6 = v5;
      if ( !v5 )
        BUG();
      v3 = 0;
      if ( *(_BYTE *)(v5 - 24) != 95 )
        return v3;
      v3 = *(_BYTE *)(v5 - 22) & 1;
      if ( (*(_BYTE *)(v5 - 22) & 1) != 0 )
        return a2;
      v7 = *(_DWORD *)(v5 - 20) & 0x7FFFFFF;
      if ( v7 )
      {
        v8 = v7;
        v9 = 0;
        v10 = 32 * v8;
        while ( 1 )
        {
          if ( (*(_BYTE *)(v6 - 17) & 0x40) != 0 )
          {
            v11 = *(_QWORD *)(*(_QWORD *)(v6 - 32) + v9);
            v12 = *(_QWORD *)(v11 + 8);
            if ( *(_BYTE *)(v12 + 8) != 16 )
              goto LABEL_14;
          }
          else
          {
            v11 = *(_QWORD *)(v6 - v10 + v9 - 24);
            v12 = *(_QWORD *)(v11 + 8);
            if ( *(_BYTE *)(v12 + 8) != 16 )
            {
LABEL_14:
              if ( *(_BYTE *)v11 == 20 )
                return v3;
              goto LABEL_15;
            }
          }
          if ( !*(_QWORD *)(v12 + 32) )
            return v3;
LABEL_15:
          v9 += 32;
          if ( v10 == v9 )
            return 1;
        }
      }
      return 1;
    }
    return 0;
  }
  return 1;
}
