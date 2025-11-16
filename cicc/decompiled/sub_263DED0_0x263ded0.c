// Function: sub_263DED0
// Address: 0x263ded0
//
__int64 __fastcall sub_263DED0(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 v4; // rsi
  unsigned int v5; // eax
  __int64 *v6; // rcx
  __int64 v7; // rdi
  int v9; // ecx
  int v10; // r10d

  v2 = *(unsigned int *)(a1 + 216);
  v4 = *(_QWORD *)(a1 + 200);
  if ( (_DWORD)v2 )
  {
    v5 = (v2 - 1) & (((0xBF58476D1CE4E5B9LL * a2) >> 31) ^ (484763065 * a2));
    v6 = (__int64 *)(v4 + 16LL * v5);
    v7 = *v6;
    if ( *v6 == a2 )
    {
LABEL_3:
      if ( v6 != (__int64 *)(v4 + 16 * v2) )
        return v6[1];
    }
    else
    {
      v9 = 1;
      while ( v7 != -1 )
      {
        v10 = v9 + 1;
        v5 = (v2 - 1) & (v9 + v5);
        v6 = (__int64 *)(v4 + 16LL * v5);
        v7 = *v6;
        if ( *v6 == a2 )
          goto LABEL_3;
        v9 = v10;
      }
    }
  }
  return 0;
}
