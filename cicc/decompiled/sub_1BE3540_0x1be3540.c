// Function: sub_1BE3540
// Address: 0x1be3540
//
__int64 __fastcall sub_1BE3540(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rdx
  __int64 v7; // r12
  _BYTE *v8; // rax
  char *v9; // r14
  size_t v10; // rax
  _DWORD *v11; // rcx
  size_t v12; // rdx
  __int64 v13; // rdx
  unsigned __int64 v15; // rdi
  char *v16; // rcx
  char *v17; // r14
  unsigned int v18; // ecx
  unsigned int v19; // ecx
  unsigned int v20; // eax
  __int64 v21; // rsi

  v5 = *(_QWORD *)(a2 + 24);
  if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v5) <= 2 )
  {
    v7 = sub_16E7EE0(a2, " +\n", 3u);
  }
  else
  {
    *(_BYTE *)(v5 + 2) = 10;
    v7 = a2;
    *(_WORD *)v5 = 11040;
    *(_QWORD *)(a2 + 24) += 3LL;
  }
  sub_16E2CE0(a3, v7);
  v8 = *(_BYTE **)(v7 + 24);
  if ( *(_BYTE **)(v7 + 16) == v8 )
  {
    v7 = sub_16E7EE0(v7, "\"", 1u);
  }
  else
  {
    *v8 = 34;
    ++*(_QWORD *)(v7 + 24);
  }
  v9 = "CLONE ";
  if ( !*(_BYTE *)(a1 + 48) )
    v9 = "REPLICATE ";
  v10 = strlen(v9);
  v11 = *(_DWORD **)(v7 + 24);
  v12 = v10;
  if ( v10 <= *(_QWORD *)(v7 + 16) - (_QWORD)v11 )
  {
    if ( (unsigned int)v10 < 8 )
    {
      if ( (v10 & 4) != 0 )
      {
        *v11 = *(_DWORD *)v9;
        *(_DWORD *)((char *)v11 + (unsigned int)v10 - 4) = *(_DWORD *)&v9[(unsigned int)v10 - 4];
      }
      else if ( (_DWORD)v10 )
      {
        *(_BYTE *)v11 = *v9;
        if ( (v10 & 2) != 0 )
          *(_WORD *)((char *)v11 + (unsigned int)v10 - 2) = *(_WORD *)&v9[(unsigned int)v10 - 2];
      }
    }
    else
    {
      v15 = (unsigned __int64)(v11 + 2) & 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)v11 = *(_QWORD *)v9;
      *(_QWORD *)((char *)v11 + (unsigned int)v10 - 8) = *(_QWORD *)&v9[(unsigned int)v10 - 8];
      v16 = (char *)v11 - v15;
      v17 = (char *)(v9 - v16);
      v18 = (v10 + (_DWORD)v16) & 0xFFFFFFF8;
      if ( v18 >= 8 )
      {
        v19 = v18 & 0xFFFFFFF8;
        v20 = 0;
        do
        {
          v21 = v20;
          v20 += 8;
          *(_QWORD *)(v15 + v21) = *(_QWORD *)&v17[v21];
        }
        while ( v20 < v19 );
      }
    }
    *(_QWORD *)(v7 + 24) += v12;
  }
  else
  {
    v7 = sub_16E7EE0(v7, v9, v10);
  }
  sub_1BE27E0(v7, *(_QWORD *)(a1 + 40));
  v13 = *(_QWORD *)(a2 + 24);
  if ( *(_BYTE *)(a1 + 50) )
  {
    if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v13) <= 6 )
    {
      sub_16E7EE0(a2, " (S->V)", 7u);
      v13 = *(_QWORD *)(a2 + 24);
    }
    else
    {
      *(_DWORD *)v13 = 760424480;
      *(_WORD *)(v13 + 4) = 22078;
      *(_BYTE *)(v13 + 6) = 41;
      v13 = *(_QWORD *)(a2 + 24) + 7LL;
      *(_QWORD *)(a2 + 24) = v13;
    }
  }
  if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v13) <= 2 )
    return sub_16E7EE0(a2, "\\l\"", 3u);
  *(_BYTE *)(v13 + 2) = 34;
  *(_WORD *)v13 = 27740;
  *(_QWORD *)(a2 + 24) += 3LL;
  return 27740;
}
