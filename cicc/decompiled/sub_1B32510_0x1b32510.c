// Function: sub_1B32510
// Address: 0x1b32510
//
void __fastcall sub_1B32510(_QWORD *src, _QWORD *a2, __int64 a3)
{
  _QWORD *v4; // rdx
  int v6; // eax
  __int64 v7; // r8
  unsigned int v8; // r9d
  unsigned int v9; // edi
  __int64 *v10; // rsi
  __int64 v11; // rcx
  unsigned int v12; // r11d
  unsigned int v13; // edi
  __int64 *v14; // rsi
  __int64 v15; // r14
  _QWORD *v16; // r14
  int v17; // eax
  __int64 v18; // r12
  _QWORD *v19; // rsi
  int v20; // esi
  __int64 v21; // rdi
  _QWORD *i; // rsi
  unsigned int v23; // r10d
  __int64 *v24; // rdx
  unsigned int v25; // r10d
  unsigned int v26; // ecx
  __int64 *v27; // rdx
  __int64 v28; // r11
  int v29; // eax
  int v30; // edx
  int v31; // edx
  __int64 v32; // r11
  int v33; // esi
  int v34; // r11d
  int v35; // r10d
  int v36; // r15d
  int v37; // [rsp+4h] [rbp-3Ch]

  if ( src != a2 )
  {
    v4 = src + 1;
    if ( a2 != src + 1 )
    {
      while ( 1 )
      {
        while ( 1 )
        {
          v17 = *(_DWORD *)(a3 + 920);
          v18 = *v4;
          v19 = v4;
          v16 = v4 + 1;
          if ( v17 )
            break;
LABEL_12:
          *v19 = v18;
LABEL_13:
          v4 = v16;
          if ( a2 == v16 )
            return;
        }
        v6 = v17 - 1;
        v7 = *(_QWORD *)(a3 + 904);
        v8 = ((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4);
        v9 = v6 & v8;
        v10 = (__int64 *)(v7 + 16LL * (v6 & v8));
        v11 = *v10;
        if ( v18 == *v10 )
        {
LABEL_5:
          v12 = *((_DWORD *)v10 + 2);
        }
        else
        {
          v32 = *v10;
          v33 = 1;
          while ( v32 != -8 )
          {
            v35 = v33 + 1;
            v9 = v6 & (v33 + v9);
            v10 = (__int64 *)(v7 + 16LL * v9);
            v32 = *v10;
            if ( v18 == *v10 )
              goto LABEL_5;
            v33 = v35;
          }
          v12 = 0;
        }
        v13 = v6 & (((unsigned int)*src >> 9) ^ ((unsigned int)*src >> 4));
        v14 = (__int64 *)(v7 + 16LL * v13);
        v15 = *v14;
        if ( *src != *v14 )
          break;
LABEL_7:
        v16 = v4 + 1;
        if ( *((_DWORD *)v14 + 2) <= v12 )
          goto LABEL_18;
        if ( src != v4 )
          memmove(src + 1, src, (char *)v4 - (char *)src);
        *src = v18;
        v4 = v16;
        if ( a2 == v16 )
          return;
      }
      v20 = 1;
      while ( v15 != -8 )
      {
        v36 = v20 + 1;
        v13 = v6 & (v20 + v13);
        v14 = (__int64 *)(v7 + 16LL * v13);
        v15 = *v14;
        if ( *src == *v14 )
          goto LABEL_7;
        v20 = v36;
      }
      v16 = v4 + 1;
LABEL_18:
      v21 = *(v4 - 1);
      for ( i = v4 - 1; ; --i )
      {
        v23 = v6 & v8;
        v24 = (__int64 *)(v7 + 16LL * (v6 & v8));
        if ( v18 == v11 )
        {
LABEL_21:
          v25 = *((_DWORD *)v24 + 2);
        }
        else
        {
          v30 = 1;
          while ( v11 != -8 )
          {
            v34 = v30 + 1;
            v23 = v6 & (v30 + v23);
            v24 = (__int64 *)(v7 + 16LL * v23);
            v11 = *v24;
            if ( v18 == *v24 )
              goto LABEL_21;
            v30 = v34;
          }
          v25 = 0;
        }
        v26 = v6 & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
        v27 = (__int64 *)(v7 + 16LL * v26);
        v28 = *v27;
        if ( *v27 != v21 )
          break;
LABEL_23:
        if ( *((_DWORD *)v27 + 2) <= v25 )
          goto LABEL_31;
        i[1] = v21;
        v29 = *(_DWORD *)(a3 + 920);
        if ( !v29 )
        {
          *i = v18;
          goto LABEL_13;
        }
        v6 = v29 - 1;
        v7 = *(_QWORD *)(a3 + 904);
        v21 = *(i - 1);
        v11 = *(_QWORD *)(v7 + 16LL * (v8 & v6));
      }
      v31 = 1;
      while ( v28 != -8 )
      {
        v26 = v6 & (v31 + v26);
        v37 = v31 + 1;
        v27 = (__int64 *)(v7 + 16LL * v26);
        v28 = *v27;
        if ( *v27 == v21 )
          goto LABEL_23;
        v31 = v37;
      }
LABEL_31:
      v19 = i + 1;
      goto LABEL_12;
    }
  }
}
