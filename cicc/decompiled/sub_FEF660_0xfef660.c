// Function: sub_FEF660
// Address: 0xfef660
//
__int64 __fastcall sub_FEF660(__int64 a1, __int64 *a2)
{
  char v2; // dl
  __int64 v3; // rcx
  int v4; // r8d
  int v5; // r9d
  __int64 v6; // rsi
  int v7; // ebx
  unsigned int i; // eax
  __int64 v9; // r10
  unsigned int v10; // eax
  __int64 v11; // r8
  __int64 v12; // rax
  __int64 v14; // r10
  __int64 v15; // [rsp+0h] [rbp-10h]

  v2 = *(_BYTE *)(a1 + 176) & 1;
  if ( v2 )
  {
    v3 = a1 + 184;
    v4 = 3;
  }
  else
  {
    v11 = *(unsigned int *)(a1 + 192);
    v3 = *(_QWORD *)(a1 + 184);
    if ( !(_DWORD)v11 )
    {
LABEL_17:
      v14 = 24 * v11;
LABEL_18:
      v9 = v3 + v14;
      goto LABEL_10;
    }
    v4 = v11 - 1;
  }
  v5 = *((_DWORD *)a2 + 2);
  v6 = *a2;
  v7 = 1;
  for ( i = v4
          & (((0xBF58476D1CE4E5B9LL
             * ((unsigned int)(37 * v5) | ((unsigned __int64)(((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4)) << 32))) >> 31)
           ^ (756364221 * v5)); ; i = v4 & v10 )
  {
    v9 = v3 + 24LL * i;
    if ( *(_QWORD *)v9 == v6 && *(_DWORD *)(v9 + 8) == v5 )
      break;
    if ( *(_QWORD *)v9 == -4096 && *(_DWORD *)(v9 + 8) == 0x7FFFFFFF )
    {
      if ( !v2 )
      {
        v11 = *(unsigned int *)(a1 + 192);
        goto LABEL_17;
      }
      v14 = 96;
      goto LABEL_18;
    }
    v10 = v7 + i;
    ++v7;
  }
LABEL_10:
  v12 = 96;
  if ( !v2 )
    v12 = 24LL * *(unsigned int *)(a1 + 192);
  if ( v9 == v3 + v12 )
  {
    BYTE4(v15) = 0;
  }
  else
  {
    BYTE4(v15) = 1;
    LODWORD(v15) = *(_DWORD *)(v9 + 16);
  }
  return v15;
}
