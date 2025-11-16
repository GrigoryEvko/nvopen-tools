// Function: sub_2B23C00
// Address: 0x2b23c00
//
__int64 *__fastcall sub_2B23C00(__int64 *a1, signed int a2, __int64 a3, __int64 a4, int a5)
{
  __int64 v10; // rax
  unsigned __int64 v11; // rsi
  __int64 v12; // rdx
  signed int v13; // edx
  unsigned __int64 v14; // r11
  unsigned int v15; // edx
  unsigned __int64 v17; // rdx
  unsigned __int64 v18; // rdi

  sub_B48880(a1, a2, 1u);
  if ( a3 + 4 * a4 != a3 )
  {
    v10 = 0;
    v11 = (unsigned __int64)(4 * a4 - 4) >> 2;
    while ( 1 )
    {
      v13 = *(_DWORD *)(a3 + 4 * v10);
      if ( v13 == -1 )
        break;
      if ( !a5 )
      {
        if ( v13 < a2 )
        {
          v18 = *a1;
          if ( (*a1 & 1) != 0 )
            *a1 = 2 * ((v18 >> 58 << 57) | ~(1LL << v13) & ~(-1LL << (v18 >> 58)) & (v18 >> 1)) + 1;
          else
            *(_QWORD *)(*(_QWORD *)v18 + 8LL * ((unsigned int)v13 >> 6)) &= ~(1LL << v13);
        }
        goto LABEL_4;
      }
      if ( a5 != 1 || v13 < a2 )
        goto LABEL_4;
      v14 = *a1;
      v15 = v13 - a2;
      if ( (*a1 & 1) == 0 )
      {
        *(_QWORD *)(*(_QWORD *)v14 + 8LL * (v15 >> 6)) &= ~(1LL << v15);
        goto LABEL_4;
      }
      *a1 = 2
          * (((unsigned __int64)*a1 >> 58 << 57) | ~(1LL << v15) & (v14 >> 1) & ~(-1LL << ((unsigned __int64)*a1 >> 58)))
          + 1;
      v12 = v10 + 1;
      if ( v11 == v10 )
        return a1;
LABEL_5:
      v10 = v12;
    }
    if ( a5 == 2 )
    {
      v17 = *a1;
      if ( (*a1 & 1) != 0 )
        *a1 = 2 * ((v17 >> 58 << 57) | ~(1LL << v10) & ~(-1LL << (v17 >> 58)) & (v17 >> 1)) + 1;
      else
        *(_QWORD *)(*(_QWORD *)v17 + 8LL * ((unsigned int)v10 >> 6)) &= ~(1LL << v10);
    }
LABEL_4:
    v12 = v10 + 1;
    if ( v11 == v10 )
      return a1;
    goto LABEL_5;
  }
  return a1;
}
