// Function: sub_271D660
// Address: 0x271d660
//
__int64 __fastcall sub_271D660(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v3; // r14d
  __int64 v5; // rsi
  __int64 v6; // rbx
  __int64 *v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  char v11; // al
  __int64 *v12; // rax
  __int64 *v14; // rax
  unsigned int v15; // eax

  LOBYTE(v3) = *(_BYTE *)(a1 + 2) == 5;
  if ( *(_BYTE *)(a2 + 12) )
  {
    v5 = *(unsigned int *)(a2 + 8);
  }
  else
  {
    v15 = sub_B6ED60(**(__int64 ***)a2, "clang.imprecise_release", 0x17u);
    *(_BYTE *)(a2 + 12) = 1;
    *(_DWORD *)(a2 + 8) = v15;
    v5 = v15;
  }
  if ( (_DWORD)v5 )
  {
    if ( (*(_BYTE *)(a3 + 7) & 0x20) != 0 )
    {
      v6 = sub_B91C10(a3, v5);
      if ( v6 )
        goto LABEL_5;
    }
  }
  else
  {
    v6 = *(_QWORD *)(a3 + 48);
    if ( v6 )
    {
LABEL_5:
      sub_271D520(a1, 5);
      goto LABEL_6;
    }
  }
  sub_271D520(a1, 4);
  if ( !*(_BYTE *)(a1 + 100) )
  {
LABEL_25:
    v6 = 0;
    sub_C8CC70(a1 + 72, a3, (__int64)v7, v8, v9, v10);
    goto LABEL_6;
  }
  v14 = *(__int64 **)(a1 + 80);
  v8 = *(unsigned int *)(a1 + 92);
  v7 = &v14[v8];
  if ( v14 == v7 )
  {
LABEL_18:
    if ( (unsigned int)v8 < *(_DWORD *)(a1 + 88) )
    {
      v8 = (unsigned int)(v8 + 1);
      v6 = 0;
      *(_DWORD *)(a1 + 92) = v8;
      *v7 = a3;
      ++*(_QWORD *)(a1 + 72);
      goto LABEL_6;
    }
    goto LABEL_25;
  }
  while ( a3 != *v14 )
  {
    if ( v7 == ++v14 )
      goto LABEL_18;
  }
  v6 = 0;
LABEL_6:
  v11 = *(_BYTE *)a1;
  *(_QWORD *)(a1 + 16) = v6;
  *(_BYTE *)(a1 + 8) = v11;
  *(_BYTE *)(a1 + 9) = (*(_WORD *)(a3 + 2) & 3u) - 1 <= 1;
  if ( !*(_BYTE *)(a1 + 52) )
  {
LABEL_13:
    sub_C8CC70(a1 + 24, a3, (__int64)v7, v8, v9, v10);
    goto LABEL_11;
  }
  v12 = *(__int64 **)(a1 + 32);
  v8 = *(unsigned int *)(a1 + 44);
  v7 = &v12[v8];
  if ( v12 == v7 )
  {
LABEL_12:
    if ( (unsigned int)v8 < *(_DWORD *)(a1 + 40) )
    {
      *(_DWORD *)(a1 + 44) = v8 + 1;
      *v7 = a3;
      ++*(_QWORD *)(a1 + 24);
      goto LABEL_11;
    }
    goto LABEL_13;
  }
  while ( a3 != *v12 )
  {
    if ( v7 == ++v12 )
      goto LABEL_12;
  }
LABEL_11:
  sub_271D2C0((_BYTE *)a1);
  return v3;
}
