// Function: sub_2EB7F10
// Address: 0x2eb7f10
//
void __fastcall sub_2EB7F10(__int64 *src, __int64 *a2, __int64 a3)
{
  __int64 *i; // r12
  __int64 v6; // r14
  unsigned int v7; // r13d
  bool v8; // cf
  __int64 v9; // r13
  __int64 *j; // rsi
  int v11; // r9d
  unsigned int v12; // edx
  __int64 *v13; // rax
  __int64 v14; // r11
  unsigned int v15; // r11d
  unsigned int v16; // edx
  __int64 *v17; // rax
  __int64 v18; // r10
  unsigned int v19; // eax
  __int64 v20; // rdi
  __int64 v21; // rcx
  __int64 v22; // r8
  int v23; // eax
  int v24; // eax
  int v25; // r10d
  int v26; // [rsp+4h] [rbp-8Ch]
  __int64 *v28[4]; // [rsp+20h] [rbp-70h] BYREF
  __int64 *v29[10]; // [rsp+40h] [rbp-50h] BYREF

  if ( src != a2 )
  {
    for ( i = src + 1; a2 != i; ++i )
    {
      while ( 1 )
      {
        v6 = *src;
        sub_2E6E850(v28, (__int64 *)a3, *i);
        v7 = *((_DWORD *)v28[2] + 2);
        sub_2E6E850(v29, (__int64 *)a3, v6);
        v8 = v7 < *((_DWORD *)v29[2] + 2);
        v9 = *i;
        if ( !v8 )
          break;
        if ( src != i )
          memmove(src + 1, src, (char *)i - (char *)src);
        *src = v9;
        if ( a2 == ++i )
          return;
      }
      for ( j = i; ; --j )
      {
        v20 = *(unsigned int *)(a3 + 24);
        v21 = *(j - 1);
        v22 = *(_QWORD *)(a3 + 8);
        if ( !(_DWORD)v20 )
          break;
        v11 = v20 - 1;
        v12 = (v20 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
        v13 = (__int64 *)(v22 + 16LL * v12);
        v14 = *v13;
        if ( v9 == *v13 )
        {
LABEL_10:
          v15 = *((_DWORD *)v13 + 2);
        }
        else
        {
          v24 = 1;
          while ( v14 != -4096 )
          {
            v25 = v24 + 1;
            v12 = v11 & (v24 + v12);
            v13 = (__int64 *)(v22 + 16LL * v12);
            v14 = *v13;
            if ( v9 == *v13 )
              goto LABEL_10;
            v24 = v25;
          }
          v15 = *(_DWORD *)(v22 + 16LL * (unsigned int)v20 + 8);
        }
        v16 = v11 & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
        v17 = (__int64 *)(v22 + 16LL * v16);
        v18 = *v17;
        if ( *v17 == v21 )
        {
LABEL_12:
          v19 = *((_DWORD *)v17 + 2);
        }
        else
        {
          v23 = 1;
          while ( v18 != -4096 )
          {
            v16 = v11 & (v23 + v16);
            v26 = v23 + 1;
            v17 = (__int64 *)(v22 + 16LL * v16);
            v18 = *v17;
            if ( v21 == *v17 )
              goto LABEL_12;
            v23 = v26;
          }
          v19 = *(_DWORD *)(v22 + 16 * v20 + 8);
        }
        if ( v19 <= v15 )
          break;
        *j = v21;
      }
      *j = v9;
    }
  }
}
