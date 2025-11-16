// Function: sub_25F8290
// Address: 0x25f8290
//
void __fastcall sub_25F8290(char *src, char *a2)
{
  char *i; // r13
  bool v5; // dl
  char *v6; // rdi
  __int64 v7; // rdx
  int v8; // ecx
  __int64 v9; // rbx
  __int64 v10; // rax
  __int64 v11; // rdx
  bool v12; // of
  signed __int64 v13; // rax
  signed __int64 v14; // rdx
  __int64 v15; // rax
  int v16; // esi
  int v17; // r9d
  int v18; // r10d
  __int64 v19; // r8
  signed __int64 v20; // rdi
  __int64 v22; // rcx
  int v23; // r9d
  __int64 v24; // r11
  __int64 v25; // rdx
  signed __int64 v26; // rdx
  signed __int64 v27; // rax
  bool v28; // cc

  if ( src != a2 )
  {
    for ( i = src + 8; i != a2; *(_QWORD *)v6 = v9 )
    {
      while ( 1 )
      {
        v7 = *(_QWORD *)src;
        v8 = 1;
        v9 = *(_QWORD *)i;
        v10 = *(_QWORD *)(*(_QWORD *)src + 280LL);
        if ( *(_DWORD *)(*(_QWORD *)src + 304LL) != 1 )
          v8 = *(_DWORD *)(v7 + 288);
        v11 = *(_QWORD *)(v7 + 296);
        v12 = __OFSUB__(v10, v11);
        v13 = v10 - v11;
        if ( v12 )
        {
          v28 = v11 <= 0;
          v14 = 0x7FFFFFFFFFFFFFFFLL;
          if ( !v28 )
            v14 = 0x8000000000000000LL;
        }
        else
        {
          v14 = v13;
        }
        v15 = *(_QWORD *)(v9 + 280);
        v16 = *(_DWORD *)(v9 + 288);
        v17 = 1;
        v18 = *(_DWORD *)(v9 + 304);
        v19 = *(_QWORD *)(v9 + 296);
        if ( v18 != 1 )
          v17 = *(_DWORD *)(v9 + 288);
        v20 = v15 - v19;
        if ( __OFSUB__(v15, v19) )
        {
          v20 = 0x8000000000000000LL;
          if ( v19 <= 0 )
            v20 = 0x7FFFFFFFFFFFFFFFLL;
        }
        v5 = v17 == v8 ? v20 > v14 : v8 < v17;
        v6 = i;
        if ( !v5 )
          break;
        if ( src != i )
          memmove(src + 8, src, i - src);
        i += 8;
        *(_QWORD *)src = v9;
        if ( i == a2 )
          return;
      }
      while ( 1 )
      {
        v22 = *((_QWORD *)v6 - 1);
        v23 = 1;
        v24 = *(_QWORD *)(v22 + 296);
        v25 = *(_QWORD *)(v22 + 280);
        if ( *(_DWORD *)(v22 + 304) != 1 )
          v23 = *(_DWORD *)(v22 + 288);
        v12 = __OFSUB__(v25, v24);
        v26 = v25 - v24;
        if ( v12 )
        {
          v26 = 0x7FFFFFFFFFFFFFFFLL;
          if ( v24 > 0 )
            v26 = 0x8000000000000000LL;
        }
        if ( v18 == 1 )
          v16 = 1;
        v12 = __OFSUB__(v15, v19);
        v27 = v15 - v19;
        if ( v12 )
        {
          v27 = 0x7FFFFFFFFFFFFFFFLL;
          if ( v19 > 0 )
            v27 = 0x8000000000000000LL;
        }
        if ( !(v16 == v23 ? v27 > v26 : v23 < v16) )
          break;
        *(_QWORD *)v6 = v22;
        v15 = *(_QWORD *)(v9 + 280);
        v6 -= 8;
        v16 = *(_DWORD *)(v9 + 288);
        v18 = *(_DWORD *)(v9 + 304);
        v19 = *(_QWORD *)(v9 + 296);
      }
      i += 8;
    }
  }
}
