// Function: sub_18DC190
// Address: 0x18dc190
//
__int64 __fastcall sub_18DC190(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5)
{
  unsigned int v7; // r12d
  __int64 *v9; // rax
  __int64 *v10; // rsi
  unsigned int v11; // edi
  __int64 *v12; // rcx

  v7 = a5;
  LOBYTE(v7) = (sub_18DC2F0(a2, a3, a4, a5) ^ 1) & (a5 != 20);
  if ( !(_BYTE)v7 )
  {
    sub_18DB880((_BYTE *)a1);
    if ( *(_BYTE *)(a1 + 2) != 1 )
      return v7;
    sub_18DB890(a1, 2);
    v9 = *(__int64 **)(a1 + 88);
    if ( *(__int64 **)(a1 + 96) != v9 )
      goto LABEL_5;
    v10 = &v9[*(unsigned int *)(a1 + 108)];
    v11 = *(_DWORD *)(a1 + 108);
    if ( v9 != v10 )
    {
      v12 = 0;
      while ( a2 != *v9 )
      {
        if ( *v9 == -2 )
          v12 = v9;
        if ( v10 == ++v9 )
        {
          if ( !v12 )
            goto LABEL_16;
          *v12 = a2;
          --*(_DWORD *)(a1 + 112);
          ++*(_QWORD *)(a1 + 80);
          return 1;
        }
      }
      return 1;
    }
LABEL_16:
    if ( v11 < *(_DWORD *)(a1 + 104) )
    {
      *(_DWORD *)(a1 + 108) = v11 + 1;
      *v10 = a2;
      ++*(_QWORD *)(a1 + 80);
    }
    else
    {
LABEL_5:
      sub_16CCBA0(a1 + 80, a2);
    }
    return 1;
  }
  return 0;
}
