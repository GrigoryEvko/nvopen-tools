// Function: sub_918D80
// Address: 0x918d80
//
__int64 __fastcall sub_918D80(__int64 a1, __int64 a2)
{
  __int64 v3; // rbx
  char i; // al
  unsigned int v5; // esi
  __int64 v6; // rdi
  unsigned int v7; // ecx
  __int64 *v8; // rdx
  __int64 v9; // r9
  unsigned int v10; // r13d
  int v12; // edx
  int v13; // r15d
  unsigned int v14; // ecx
  _QWORD *v15; // rdx
  _QWORD *v16; // rax
  __int64 v17; // r11
  _BYTE *v18; // rax
  unsigned int v19; // eax
  __int64 v20; // r14
  int v21; // r8d
  int v22; // r8d
  __int64 v23; // r10
  unsigned int v24; // ecx
  int v25; // edx
  __int64 v26; // r9
  int v27; // r10d
  int v28; // edi
  int v29; // r8d
  int v30; // r8d
  __int64 v31; // r9
  int v32; // ecx
  __int64 v33; // r14
  _QWORD *v34; // rdi
  __int64 v35; // rsi
  int v36; // edi
  _QWORD *v37; // rsi

  v3 = a2;
  for ( i = *(_BYTE *)(a2 + 140); i == 12; i = *(_BYTE *)(v3 + 140) )
    v3 = *(_QWORD *)(v3 + 160);
  v5 = *(_DWORD *)(a1 + 320);
  v6 = *(_QWORD *)(a1 + 304);
  if ( !v5 )
    goto LABEL_10;
  v7 = (v5 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
  v8 = (__int64 *)(v6 + 16LL * v7);
  v9 = *v8;
  if ( *v8 != v3 )
  {
    v12 = 1;
    while ( v9 != -4096 )
    {
      v27 = v12 + 1;
      v7 = (v5 - 1) & (v12 + v7);
      v8 = (__int64 *)(v6 + 16LL * v7);
      v9 = *v8;
      if ( *v8 == v3 )
        goto LABEL_5;
      v12 = v27;
    }
LABEL_10:
    if ( i == 10 )
    {
      v20 = *(_QWORD *)(v3 + 160);
      if ( v20 )
      {
        do
        {
          v10 = sub_918D80(a1, *(_QWORD *)(v20 + 120));
          if ( (_BYTE)v10 )
            break;
          v20 = *(_QWORD *)(v20 + 112);
        }
        while ( v20 );
        v6 = *(_QWORD *)(a1 + 304);
        v5 = *(_DWORD *)(a1 + 320);
      }
      else
      {
        v10 = 0;
      }
    }
    else
    {
      v10 = 1;
      if ( i != 11 )
      {
        v10 = 0;
        if ( i == 8 )
        {
          v19 = sub_918D80(a1, *(_QWORD *)(v3 + 160));
          v6 = *(_QWORD *)(a1 + 304);
          v5 = *(_DWORD *)(a1 + 320);
          v10 = v19;
        }
      }
    }
    if ( v5 )
    {
      v13 = 1;
      v14 = (v5 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
      v15 = (_QWORD *)(v6 + 16LL * v14);
      v16 = 0;
      v17 = *v15;
      if ( *v15 == v3 )
      {
LABEL_16:
        v18 = v15 + 1;
LABEL_17:
        *v18 = v10;
        return v10;
      }
      while ( v17 != -4096 )
      {
        if ( !v16 && v17 == -8192 )
          v16 = v15;
        v14 = (v5 - 1) & (v13 + v14);
        v15 = (_QWORD *)(v6 + 16LL * v14);
        v17 = *v15;
        if ( *v15 == v3 )
          goto LABEL_16;
        ++v13;
      }
      v28 = *(_DWORD *)(a1 + 312);
      if ( !v16 )
        v16 = v15;
      ++*(_QWORD *)(a1 + 296);
      v25 = v28 + 1;
      if ( 4 * (v28 + 1) < 3 * v5 )
      {
        if ( v5 - (v25 + *(_DWORD *)(a1 + 316)) > v5 >> 3 )
        {
LABEL_25:
          *(_DWORD *)(a1 + 312) = v25;
          if ( *v16 != -4096 )
            --*(_DWORD *)(a1 + 316);
          *v16 = v3;
          v18 = v16 + 1;
          *v18 = 0;
          goto LABEL_17;
        }
        sub_918BA0(a1 + 296, v5);
        v29 = *(_DWORD *)(a1 + 320);
        if ( v29 )
        {
          v30 = v29 - 1;
          v31 = *(_QWORD *)(a1 + 304);
          v32 = 1;
          LODWORD(v33) = v30 & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
          v25 = *(_DWORD *)(a1 + 312) + 1;
          v34 = 0;
          v16 = (_QWORD *)(v31 + 16LL * (unsigned int)v33);
          v35 = *v16;
          if ( *v16 != v3 )
          {
            while ( v35 != -4096 )
            {
              if ( !v34 && v35 == -8192 )
                v34 = v16;
              v33 = v30 & (unsigned int)(v33 + v32);
              v16 = (_QWORD *)(v31 + 16 * v33);
              v35 = *v16;
              if ( v3 == *v16 )
                goto LABEL_25;
              ++v32;
            }
            if ( v34 )
              v16 = v34;
          }
          goto LABEL_25;
        }
LABEL_62:
        ++*(_DWORD *)(a1 + 312);
        BUG();
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 296);
    }
    sub_918BA0(a1 + 296, 2 * v5);
    v21 = *(_DWORD *)(a1 + 320);
    if ( v21 )
    {
      v22 = v21 - 1;
      v23 = *(_QWORD *)(a1 + 304);
      v24 = v22 & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
      v25 = *(_DWORD *)(a1 + 312) + 1;
      v16 = (_QWORD *)(v23 + 16LL * v24);
      v26 = *v16;
      if ( v3 != *v16 )
      {
        v36 = 1;
        v37 = 0;
        while ( v26 != -4096 )
        {
          if ( v26 == -8192 && !v37 )
            v37 = v16;
          v24 = v22 & (v36 + v24);
          v16 = (_QWORD *)(v23 + 16LL * v24);
          v26 = *v16;
          if ( *v16 == v3 )
            goto LABEL_25;
          ++v36;
        }
        if ( v37 )
          v16 = v37;
      }
      goto LABEL_25;
    }
    goto LABEL_62;
  }
LABEL_5:
  if ( v8 == (__int64 *)(v6 + 16LL * v5) )
    goto LABEL_10;
  return *((unsigned __int8 *)v8 + 8);
}
