// Function: sub_37B8B00
// Address: 0x37b8b00
//
void __fastcall sub_37B8B00(char *src, char *a2, __int64 a3)
{
  char *i; // r13
  __int64 v6; // rsi
  __int64 v7; // rcx
  __int64 v8; // r12
  __int64 v9; // r8
  int v10; // edi
  unsigned int v11; // edx
  __int64 *v12; // rax
  __int64 v13; // r10
  unsigned int v14; // r9d
  unsigned int v15; // edx
  __int64 *v16; // rax
  __int64 v17; // r11
  unsigned int v18; // eax
  char *j; // rdi
  int v20; // r9d
  unsigned int v21; // edx
  __int64 *v22; // rax
  __int64 v23; // r11
  unsigned int v24; // r11d
  unsigned int v25; // edx
  __int64 *v26; // rax
  __int64 v27; // r10
  unsigned int v28; // eax
  __int64 v29; // rcx
  int v30; // eax
  int v31; // eax
  int v32; // eax
  int v33; // eax
  int v34; // r10d
  int v35; // r10d
  int v36; // r9d
  int v37; // [rsp+4h] [rbp-3Ch]

  if ( src != a2 )
  {
    for ( i = src + 8; a2 != i; i += 8 )
    {
      while ( 1 )
      {
        v6 = *(unsigned int *)(a3 + 688);
        v7 = *(_QWORD *)src;
        v8 = *(_QWORD *)i;
        v9 = *(_QWORD *)(a3 + 672);
        if ( !(_DWORD)v6 )
          break;
        v10 = v6 - 1;
        v11 = (v6 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
        v12 = (__int64 *)(v9 + 16LL * v11);
        v13 = *v12;
        if ( v8 == *v12 )
        {
LABEL_9:
          v14 = *((_DWORD *)v12 + 2);
        }
        else
        {
          v33 = 1;
          while ( v13 != -4096 )
          {
            v36 = v33 + 1;
            v11 = v10 & (v33 + v11);
            v12 = (__int64 *)(v9 + 16LL * v11);
            v13 = *v12;
            if ( v8 == *v12 )
              goto LABEL_9;
            v33 = v36;
          }
          v14 = *(_DWORD *)(v9 + 16LL * (unsigned int)v6 + 8);
        }
        v15 = v10 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
        v16 = (__int64 *)(v9 + 16LL * v15);
        v17 = *v16;
        if ( v7 == *v16 )
        {
LABEL_11:
          v18 = *((_DWORD *)v16 + 2);
        }
        else
        {
          v32 = 1;
          while ( v17 != -4096 )
          {
            v35 = v32 + 1;
            v15 = v10 & (v32 + v15);
            v16 = (__int64 *)(v9 + 16LL * v15);
            v17 = *v16;
            if ( v7 == *v16 )
              goto LABEL_11;
            v32 = v35;
          }
          v18 = *(_DWORD *)(v9 + 16LL * (unsigned int)v6 + 8);
        }
        if ( v14 >= v18 )
          break;
        if ( src != i )
          memmove(src + 8, src, i - src);
        *(_QWORD *)src = v8;
        i += 8;
        if ( a2 == i )
          return;
      }
      for ( j = i; ; j -= 8 )
      {
        v29 = *((_QWORD *)j - 1);
        if ( !(_DWORD)v6 )
          break;
        v20 = v6 - 1;
        v21 = (v6 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
        v22 = (__int64 *)(v9 + 16LL * v21);
        v23 = *v22;
        if ( v8 == *v22 )
        {
LABEL_15:
          v24 = *((_DWORD *)v22 + 2);
        }
        else
        {
          v31 = 1;
          while ( v23 != -4096 )
          {
            v34 = v31 + 1;
            v21 = v20 & (v31 + v21);
            v22 = (__int64 *)(v9 + 16LL * v21);
            v23 = *v22;
            if ( v8 == *v22 )
              goto LABEL_15;
            v31 = v34;
          }
          v24 = *(_DWORD *)(v9 + 16LL * (unsigned int)v6 + 8);
        }
        v25 = v20 & (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4));
        v26 = (__int64 *)(v9 + 16LL * v25);
        v27 = *v26;
        if ( *v26 == v29 )
        {
LABEL_17:
          v28 = *((_DWORD *)v26 + 2);
        }
        else
        {
          v30 = 1;
          while ( v27 != -4096 )
          {
            v25 = v20 & (v30 + v25);
            v37 = v30 + 1;
            v26 = (__int64 *)(v9 + 16LL * v25);
            v27 = *v26;
            if ( v29 == *v26 )
              goto LABEL_17;
            v30 = v37;
          }
          v28 = *(_DWORD *)(v9 + 16 * v6 + 8);
        }
        if ( v24 >= v28 )
          break;
        *(_QWORD *)j = v29;
        v9 = *(_QWORD *)(a3 + 672);
        v6 = *(unsigned int *)(a3 + 688);
      }
      *(_QWORD *)j = v8;
    }
  }
}
