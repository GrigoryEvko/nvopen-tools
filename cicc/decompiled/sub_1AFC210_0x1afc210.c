// Function: sub_1AFC210
// Address: 0x1afc210
//
void __fastcall sub_1AFC210(char *src, char *a2, __int64 a3)
{
  char *v3; // r14
  __int64 v6; // rdi
  __int64 v7; // rax
  __int64 v8; // rbx
  __int64 v9; // r8
  int v10; // r9d
  unsigned int v11; // r15d
  unsigned int v12; // ecx
  __int64 *v13; // rdx
  __int64 v14; // rsi
  __int64 *v15; // rsi
  unsigned int v16; // r11d
  unsigned int v17; // ecx
  __int64 *v18; // rdx
  __int64 v19; // r10
  char *i; // rdi
  __int64 v21; // rsi
  int v22; // r9d
  unsigned int v23; // ecx
  __int64 *v24; // rdx
  __int64 v25; // r10
  __int64 *v26; // rax
  unsigned int v27; // r11d
  unsigned int v28; // ecx
  __int64 *v29; // rdx
  __int64 v30; // r10
  char *v31; // r15
  int v32; // edx
  int v33; // edx
  int v34; // r11d
  int v35; // edx
  int v36; // edx
  int v37; // r10d
  int v38; // [rsp+4h] [rbp-3Ch]
  int v39; // [rsp+4h] [rbp-3Ch]

  if ( src != a2 )
  {
    v3 = src + 8;
    if ( src + 8 != a2 )
    {
      while ( 1 )
      {
        v6 = *(_QWORD *)src;
        v7 = *(unsigned int *)(*(_QWORD *)a3 + 48LL);
        if ( !(_DWORD)v7 )
          goto LABEL_42;
        v8 = *(_QWORD *)v3;
        v9 = *(_QWORD *)(*(_QWORD *)a3 + 32LL);
        v10 = v7 - 1;
        v11 = ((unsigned int)*(_QWORD *)v3 >> 9) ^ ((unsigned int)*(_QWORD *)v3 >> 4);
        v12 = (v7 - 1) & v11;
        v13 = (__int64 *)(v9 + 16LL * v12);
        v14 = *v13;
        if ( *(_QWORD *)v3 != *v13 )
          break;
LABEL_5:
        v15 = (__int64 *)(v9 + 16LL * (unsigned int)v7);
        if ( v15 == v13 )
          BUG();
        v16 = *(_DWORD *)(v13[1] + 16);
        v17 = v10 & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
        v18 = (__int64 *)(v9 + 16LL * v17);
        v19 = *v18;
        if ( v6 != *v18 )
        {
          v35 = 1;
          while ( v19 != -8 )
          {
            v17 = v10 & (v35 + v17);
            v39 = v35 + 1;
            v18 = (__int64 *)(v9 + 16LL * v17);
            v19 = *v18;
            if ( v6 == *v18 )
              goto LABEL_7;
            v35 = v39;
          }
LABEL_41:
          BUG();
        }
LABEL_7:
        if ( v18 == v15 )
          goto LABEL_41;
        if ( v16 > *(_DWORD *)(v18[1] + 16) )
        {
          v31 = v3 + 8;
          if ( src != v3 )
            memmove(src + 8, src, v3 - src);
          *(_QWORD *)src = v8;
        }
        else
        {
          for ( i = v3; ; i -= 8 )
          {
            v21 = *((_QWORD *)i - 1);
            if ( !(_DWORD)v7 )
              goto LABEL_44;
            v22 = v7 - 1;
            v23 = (v7 - 1) & v11;
            v24 = (__int64 *)(v9 + 16LL * v23);
            v25 = *v24;
            if ( v8 != *v24 )
            {
              v33 = 1;
              while ( v25 != -8 )
              {
                v34 = v33 + 1;
                v23 = v22 & (v33 + v23);
                v24 = (__int64 *)(v9 + 16LL * v23);
                v25 = *v24;
                if ( v8 == *v24 )
                  goto LABEL_13;
                v33 = v34;
              }
LABEL_44:
              BUG();
            }
LABEL_13:
            v26 = (__int64 *)(v9 + 16 * v7);
            if ( v24 == v26 )
              BUG();
            v27 = *(_DWORD *)(v24[1] + 16);
            v28 = v22 & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
            v29 = (__int64 *)(v9 + 16LL * v28);
            v30 = *v29;
            if ( *v29 != v21 )
            {
              v32 = 1;
              while ( v30 != -8 )
              {
                v28 = v22 & (v32 + v28);
                v38 = v32 + 1;
                v29 = (__int64 *)(v9 + 16LL * v28);
                v30 = *v29;
                if ( v21 == *v29 )
                  goto LABEL_15;
                v32 = v38;
              }
LABEL_43:
              BUG();
            }
LABEL_15:
            if ( v29 == v26 )
              goto LABEL_43;
            if ( v27 <= *(_DWORD *)(v29[1] + 16) )
              break;
            *(_QWORD *)i = v21;
            v9 = *(_QWORD *)(*(_QWORD *)a3 + 32LL);
            v7 = *(unsigned int *)(*(_QWORD *)a3 + 48LL);
          }
          *(_QWORD *)i = v8;
          v31 = v3 + 8;
        }
        v3 = v31;
        if ( a2 == v31 )
          return;
      }
      v36 = 1;
      while ( v14 != -8 )
      {
        v37 = v36 + 1;
        v12 = v10 & (v36 + v12);
        v13 = (__int64 *)(v9 + 16LL * v12);
        v14 = *v13;
        if ( v8 == *v13 )
          goto LABEL_5;
        v36 = v37;
      }
LABEL_42:
      BUG();
    }
  }
}
