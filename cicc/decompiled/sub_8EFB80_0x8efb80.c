// Function: sub_8EFB80
// Address: 0x8efb80
//
__int64 __fastcall sub_8EFB80(char *s, unsigned __int64 a2, __int64 a3)
{
  char *v4; // r12
  unsigned int v6; // eax
  unsigned int v7; // r14d
  int v9; // esi
  unsigned __int64 v10; // rax
  __int64 v11; // rdx
  char v12; // al
  __int64 v13; // rax
  char *v14; // rcx
  int v15; // edi
  _BYTE *v16; // rdx
  char v17; // cl
  int v18; // eax
  int v19; // ecx
  unsigned __int64 v20; // rax
  char sa[64]; // [rsp+0h] [rbp-40h] BYREF

  v4 = s;
  v6 = *(_DWORD *)(a3 + 4);
  if ( v6 >= 0xFFFFFFFE || v6 == 0 )
  {
    if ( a2 >= (int)(*(_DWORD *)(a3 + 48) + 2 - v6 + 1) )
    {
      v7 = 0;
      sprintf(s, "0.%s%s", &a00000000[v6 + 8], (const char *)(a3 + 8));
      return v7;
    }
    return (unsigned int)-2;
  }
  if ( v6 - 1 > 6 )
  {
    v18 = sprintf(sa, "%d", v6 - 1);
    v19 = *(_DWORD *)(a3 + 48);
    v20 = v18 + (__int64)(v19 + 3);
    if ( v19 == 1 )
    {
      v7 = v20 < a2 ? 0 : 0xFFFFFFFE;
      sprintf(s, "%c.0E%d", (unsigned int)*(char *)(a3 + 8), *(_DWORD *)(a3 + 4) - 1);
    }
    else
    {
      v7 = -2;
      if ( v20 <= a2 )
      {
        v7 = 0;
        sprintf(s, "%c.%sE%d", (unsigned int)*(char *)(a3 + 8), (const char *)(a3 + 9), *(_DWORD *)(a3 + 4) - 1);
      }
    }
    return v7;
  }
  v9 = *(_DWORD *)(a3 + 48);
  if ( (int)v6 >= v9 )
    v10 = (int)(v6 + 3);
  else
    v10 = v9 + 2;
  if ( v10 > a2 )
    return (unsigned int)-2;
  v11 = 0;
  while ( 1 )
  {
    LODWORD(v13) = v11;
    v14 = v4 + 1;
    v15 = v11 + 1;
    if ( v9 <= (int)v11 )
      break;
    v12 = *(_BYTE *)(a3 + v11++ + 8);
    *v4 = v12;
    if ( *(_DWORD *)(a3 + 4) <= (int)v11 )
    {
      LODWORD(v13) = v15;
      ++v4;
      goto LABEL_17;
    }
    v9 = *(_DWORD *)(a3 + 48);
    ++v4;
  }
  if ( (int)v11 < *(_DWORD *)(a3 + 4) )
  {
    while ( 1 )
    {
      *(v14 - 1) = 48;
      v4 = v14;
      LODWORD(v13) = v13 + 1;
      if ( *(_DWORD *)(a3 + 4) <= (int)v13 )
        break;
      ++v14;
    }
  }
LABEL_17:
  *v4 = 46;
  v16 = v4 + 1;
  if ( *(_DWORD *)(a3 + 48) > (int)v13
    || (v4[1] = 48, LODWORD(v13) = v13 + 1, v16 = v4 + 2, (int)v13 < *(_DWORD *)(a3 + 48)) )
  {
    v13 = (int)v13;
    do
    {
      v17 = *(_BYTE *)(a3 + v13 + 8);
      ++v16;
      ++v13;
      *(v16 - 1) = v17;
    }
    while ( *(_DWORD *)(a3 + 48) > (int)v13 );
  }
  *v16 = 0;
  return 0;
}
