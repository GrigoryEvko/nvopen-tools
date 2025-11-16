// Function: sub_7762B0
// Address: 0x7762b0
//
__int64 __fastcall sub_7762B0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // r8
  unsigned __int64 v4; // rsi
  char i; // dl
  unsigned __int64 v6; // r8
  __int64 v7; // rdx
  char m; // cl
  int v9; // r10d
  unsigned int n; // ecx
  _QWORD *v11; // rdx
  __int64 v12; // rsi
  __int64 v13; // r9
  __int64 v14; // rcx
  _QWORD *v15; // r8
  _QWORD *ii; // rdx
  __int64 v17; // rdx
  int v18; // r10d
  unsigned int j; // edx
  __int64 v20; // rcx
  __int64 v21; // r9
  __int64 v22; // rdx
  _QWORD *v23; // rsi
  _QWORD *k; // rax

  result = *(_QWORD *)(a2 + 8);
  v3 = *(_QWORD *)(a2 + 24);
  if ( result )
  {
    v4 = *(_QWORD *)(result + 8);
    result = *(_QWORD *)(v4 + 120);
    for ( i = *(_BYTE *)(result + 140); i == 12; i = *(_BYTE *)(result + 140) )
      result = *(_QWORD *)(result + 160);
    if ( i == 6 )
    {
      v18 = *(_DWORD *)(a1 + 8);
      for ( j = v18 & (v4 >> 3); ; j = v18 & (j + 1) )
      {
        result = *(_QWORD *)a1 + 16LL * j;
        if ( v4 == *(_QWORD *)result )
          break;
        if ( !*(_QWORD *)result )
LABEL_34:
          BUG();
      }
      v20 = *(_QWORD *)(result + 8);
      if ( (*(_BYTE *)(v20 - 9) & 1) != 0 && (*(_BYTE *)(v20 + 8) & 4) != 0 )
      {
        v21 = *(_QWORD *)(v20 + 16);
        v22 = 2;
        v23 = *(_QWORD **)v21;
        for ( k = **(_QWORD ***)v21; k; ++v22 )
        {
          v23 = k;
          k = (_QWORD *)*k;
        }
        *v23 = qword_4F08088;
        *(_BYTE *)(v20 + 8) &= ~4u;
        result = *(_QWORD *)(v21 + 24);
        qword_4F08080 += v22;
        qword_4F08088 = v21;
        *(_QWORD *)(v20 + 16) = result;
      }
    }
  }
  if ( v3 )
  {
    if ( *(_BYTE *)(v3 + 40) == 20 )
    {
      for ( result = *(_QWORD *)(v3 + 72); result; result = *(_QWORD *)result )
      {
        if ( *(_BYTE *)(result + 8) == 7 )
        {
          v6 = *(_QWORD *)(result + 16);
          v7 = *(_QWORD *)(v6 + 120);
          for ( m = *(_BYTE *)(v7 + 140); m == 12; m = *(_BYTE *)(v7 + 140) )
            v7 = *(_QWORD *)(v7 + 160);
          if ( m == 6 )
          {
            v9 = *(_DWORD *)(a1 + 8);
            for ( n = v9 & (v6 >> 3); ; n = v9 & (n + 1) )
            {
              v11 = (_QWORD *)(*(_QWORD *)a1 + 16LL * n);
              if ( v6 == *v11 )
                break;
              if ( !*v11 )
                goto LABEL_34;
            }
            v12 = v11[1];
            if ( (*(_BYTE *)(v12 - 9) & 1) != 0 && (*(_BYTE *)(v12 + 8) & 4) != 0 )
            {
              v13 = *(_QWORD *)(v12 + 16);
              v14 = 2;
              v15 = *(_QWORD **)v13;
              for ( ii = **(_QWORD ***)v13; ii; ++v14 )
              {
                v15 = ii;
                ii = (_QWORD *)*ii;
              }
              *v15 = qword_4F08088;
              *(_BYTE *)(v12 + 8) &= ~4u;
              v17 = *(_QWORD *)(v13 + 24);
              qword_4F08080 += v14;
              qword_4F08088 = v13;
              *(_QWORD *)(v12 + 16) = v17;
            }
          }
        }
      }
    }
  }
  return result;
}
