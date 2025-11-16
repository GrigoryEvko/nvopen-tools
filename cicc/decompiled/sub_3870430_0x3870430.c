// Function: sub_3870430
// Address: 0x3870430
//
__int64 __fastcall sub_3870430(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int8 v6; // cl
  __int64 *v7; // rax
  __int64 v9; // r15
  __int64 *v10; // r12
  __int64 *v11; // r15

  do
  {
    if ( (*(_DWORD *)(a3 + 20) & 0xFFFFFFF) == 0 )
      return 0;
    v6 = *(_BYTE *)(a3 + 16);
    if ( v6 == 77 || (unsigned int)v6 - 60 <= 0xC && v6 != 71 )
      return 0;
    if ( a1[26] == a4 )
    {
      v9 = 24LL * (*(_DWORD *)(a3 + 20) & 0xFFFFFFF);
      if ( (*(_BYTE *)(a3 + 23) & 0x40) != 0 )
      {
        v7 = *(__int64 **)(a3 - 8);
        v10 = v7 + 3;
        v11 = &v7[(unsigned __int64)v9 / 8];
        if ( v7 + 3 != v11 )
          goto LABEL_19;
      }
      else
      {
        v7 = (__int64 *)(a3 - v9);
        v10 = (__int64 *)(a3 - v9 + 24);
        if ( v10 != (__int64 *)a3 )
        {
          v11 = (__int64 *)a3;
LABEL_19:
          while ( *(_BYTE *)(*v10 + 16) <= 0x17u || sub_15CCEE0(*(_QWORD *)(*a1 + 56LL), *v10, a1[27]) )
          {
            v10 += 3;
            if ( v10 == v11 )
              goto LABEL_6;
          }
          return 0;
        }
      }
    }
    else
    {
LABEL_6:
      if ( (*(_BYTE *)(a3 + 23) & 0x40) != 0 )
        v7 = *(__int64 **)(a3 - 8);
      else
        v7 = (__int64 *)(a3 - 24LL * (*(_DWORD *)(a3 + 20) & 0xFFFFFFF));
    }
    a3 = *v7;
    if ( *(_BYTE *)(*v7 + 16) <= 0x17u || (unsigned __int8)sub_15F3040(*v7) || sub_15F3330(a3) )
      return 0;
  }
  while ( a2 != a3 );
  return 1;
}
