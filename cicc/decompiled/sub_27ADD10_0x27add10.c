// Function: sub_27ADD10
// Address: 0x27add10
//
void __fastcall sub_27ADD10(char *src, char *a2, __int64 a3)
{
  char *v3; // r13
  int v6; // eax
  __int64 v7; // rdi
  __int64 v8; // r12
  __int64 v9; // r8
  int v10; // eax
  unsigned int v11; // r9d
  unsigned int v12; // esi
  __int64 *v13; // rdx
  __int64 v14; // rcx
  unsigned int v15; // r11d
  unsigned int v16; // esi
  __int64 *v17; // rdx
  __int64 v18; // r10
  __int64 v19; // rdi
  char *i; // rsi
  unsigned int v21; // r10d
  __int64 *v22; // rdx
  unsigned int v23; // r11d
  unsigned int v24; // ecx
  __int64 *v25; // rdx
  __int64 v26; // r10
  int v27; // eax
  int v28; // edx
  int v29; // edx
  char *v30; // rsi
  int v31; // edx
  int v32; // r15d
  __int64 v33; // r11
  int v34; // edx
  int v35; // r11d
  int v36; // r10d
  int v37; // [rsp+4h] [rbp-3Ch]

  if ( src != a2 )
  {
    v3 = src + 8;
    if ( a2 != src + 8 )
    {
      while ( 1 )
      {
        while ( 1 )
        {
          v6 = *(_DWORD *)(a3 + 592);
          v7 = *(_QWORD *)src;
          v8 = *(_QWORD *)v3;
          v9 = *(_QWORD *)(a3 + 576);
          if ( v6 )
            break;
          v30 = v3;
          v3 += 8;
          *(_QWORD *)v30 = v8;
LABEL_20:
          if ( a2 == v3 )
            return;
        }
        v10 = v6 - 1;
        v11 = ((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4);
        v12 = v10 & v11;
        v13 = (__int64 *)(v9 + 16LL * (v10 & v11));
        v14 = *v13;
        if ( v8 == *v13 )
        {
LABEL_9:
          v15 = *((_DWORD *)v13 + 2);
        }
        else
        {
          v33 = *v13;
          v34 = 1;
          while ( v33 != -4096 )
          {
            v36 = v34 + 1;
            v12 = v10 & (v34 + v12);
            v13 = (__int64 *)(v9 + 16LL * v12);
            v33 = *v13;
            if ( v8 == *v13 )
              goto LABEL_9;
            v34 = v36;
          }
          v15 = 0;
        }
        v16 = v10 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
        v17 = (__int64 *)(v9 + 16LL * v16);
        v18 = *v17;
        if ( v7 != *v17 )
          break;
LABEL_11:
        if ( *((_DWORD *)v17 + 2) <= v15 )
          goto LABEL_12;
        if ( src != v3 )
          memmove(src + 8, src, v3 - src);
        *(_QWORD *)src = v8;
        v3 += 8;
        if ( a2 == v3 )
          return;
      }
      v31 = 1;
      while ( v18 != -4096 )
      {
        v32 = v31 + 1;
        v16 = v10 & (v31 + v16);
        v17 = (__int64 *)(v9 + 16LL * v16);
        v18 = *v17;
        if ( v7 == *v17 )
          goto LABEL_11;
        v31 = v32;
      }
LABEL_12:
      v19 = *((_QWORD *)v3 - 1);
      for ( i = v3 - 8; ; i -= 8 )
      {
        v21 = v10 & v11;
        v22 = (__int64 *)(v9 + 16LL * (v10 & v11));
        if ( v8 == v14 )
        {
LABEL_15:
          v23 = *((_DWORD *)v22 + 2);
        }
        else
        {
          v29 = 1;
          while ( v14 != -4096 )
          {
            v35 = v29 + 1;
            v21 = v10 & (v29 + v21);
            v22 = (__int64 *)(v9 + 16LL * v21);
            v14 = *v22;
            if ( v8 == *v22 )
              goto LABEL_15;
            v29 = v35;
          }
          v23 = 0;
        }
        v24 = v10 & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
        v25 = (__int64 *)(v9 + 16LL * v24);
        v26 = *v25;
        if ( v19 != *v25 )
          break;
LABEL_17:
        if ( v23 >= *((_DWORD *)v25 + 2) )
          goto LABEL_24;
        *((_QWORD *)i + 1) = v19;
        v27 = *(_DWORD *)(a3 + 592);
        v19 = *((_QWORD *)i - 1);
        v9 = *(_QWORD *)(a3 + 576);
        if ( !v27 )
        {
          *(_QWORD *)i = v8;
          v3 += 8;
          goto LABEL_20;
        }
        v10 = v27 - 1;
        v14 = *(_QWORD *)(v9 + 16LL * (v11 & v10));
      }
      v28 = 1;
      while ( v26 != -4096 )
      {
        v24 = v10 & (v28 + v24);
        v37 = v28 + 1;
        v25 = (__int64 *)(v9 + 16LL * v24);
        v26 = *v25;
        if ( v19 == *v25 )
          goto LABEL_17;
        v28 = v37;
      }
LABEL_24:
      v3 += 8;
      *((_QWORD *)i + 1) = v8;
      goto LABEL_20;
    }
  }
}
