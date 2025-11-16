// Function: sub_12789C0
// Address: 0x12789c0
//
__int64 __fastcall sub_12789C0(__int64 a1, __int64 a2)
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
  unsigned int v13; // ecx
  __int64 *v14; // rax
  __int64 v15; // rdx
  unsigned int v16; // eax
  __int64 v17; // r14
  int v18; // r9d
  int v19; // r9d
  __int64 v20; // r10
  unsigned int v21; // ecx
  int v22; // edx
  __int64 v23; // r8
  int v24; // r10d
  int v25; // r11d
  __int64 *v26; // r10
  int v27; // edi
  int v28; // r8d
  int v29; // r8d
  __int64 v30; // r9
  int v31; // ecx
  __int64 v32; // r14
  __int64 *v33; // rdi
  __int64 v34; // rsi
  int v35; // edi
  __int64 *v36; // rsi

  v3 = a2;
  for ( i = *(_BYTE *)(a2 + 140); i == 12; i = *(_BYTE *)(v3 + 140) )
    v3 = *(_QWORD *)(v3 + 160);
  v5 = *(_DWORD *)(a1 + 336);
  v6 = *(_QWORD *)(a1 + 320);
  if ( !v5 )
    goto LABEL_10;
  v7 = (v5 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
  v8 = (__int64 *)(v6 + 16LL * v7);
  v9 = *v8;
  if ( *v8 != v3 )
  {
    v12 = 1;
    while ( v9 != -8 )
    {
      v24 = v12 + 1;
      v7 = (v5 - 1) & (v12 + v7);
      v8 = (__int64 *)(v6 + 16LL * v7);
      v9 = *v8;
      if ( *v8 == v3 )
        goto LABEL_5;
      v12 = v24;
    }
LABEL_10:
    if ( i == 10 )
    {
      v17 = *(_QWORD *)(v3 + 160);
      if ( v17 )
      {
        do
        {
          v10 = sub_12789C0(a1, *(_QWORD *)(v17 + 120));
          if ( (_BYTE)v10 )
            break;
          v17 = *(_QWORD *)(v17 + 112);
        }
        while ( v17 );
        v6 = *(_QWORD *)(a1 + 320);
        v5 = *(_DWORD *)(a1 + 336);
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
          v16 = sub_12789C0(a1, *(_QWORD *)(v3 + 160));
          v6 = *(_QWORD *)(a1 + 320);
          v5 = *(_DWORD *)(a1 + 336);
          v10 = v16;
        }
      }
    }
    if ( v5 )
    {
      v13 = (v5 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
      v14 = (__int64 *)(v6 + 16LL * v13);
      v15 = *v14;
      if ( *v14 == v3 )
      {
LABEL_16:
        *((_BYTE *)v14 + 8) = v10;
        return v10;
      }
      v25 = 1;
      v26 = 0;
      while ( v15 != -8 )
      {
        if ( v15 == -16 && !v26 )
          v26 = v14;
        v13 = (v5 - 1) & (v25 + v13);
        v14 = (__int64 *)(v6 + 16LL * v13);
        v15 = *v14;
        if ( *v14 == v3 )
          goto LABEL_16;
        ++v25;
      }
      v27 = *(_DWORD *)(a1 + 328);
      if ( v26 )
        v14 = v26;
      ++*(_QWORD *)(a1 + 312);
      v22 = v27 + 1;
      if ( 4 * (v27 + 1) < 3 * v5 )
      {
        if ( v5 - (v22 + *(_DWORD *)(a1 + 332)) > v5 >> 3 )
        {
LABEL_24:
          *(_DWORD *)(a1 + 328) = v22;
          if ( *v14 != -8 )
            --*(_DWORD *)(a1 + 332);
          *v14 = v3;
          *((_BYTE *)v14 + 8) = 0;
          goto LABEL_16;
        }
        sub_1278800(a1 + 312, v5);
        v28 = *(_DWORD *)(a1 + 336);
        if ( v28 )
        {
          v29 = v28 - 1;
          v30 = *(_QWORD *)(a1 + 320);
          v31 = 1;
          LODWORD(v32) = v29 & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
          v22 = *(_DWORD *)(a1 + 328) + 1;
          v33 = 0;
          v14 = (__int64 *)(v30 + 16LL * (unsigned int)v32);
          v34 = *v14;
          if ( *v14 != v3 )
          {
            while ( v34 != -8 )
            {
              if ( !v33 && v34 == -16 )
                v33 = v14;
              v32 = v29 & (unsigned int)(v32 + v31);
              v14 = (__int64 *)(v30 + 16 * v32);
              v34 = *v14;
              if ( *v14 == v3 )
                goto LABEL_24;
              ++v31;
            }
            if ( v33 )
              v14 = v33;
          }
          goto LABEL_24;
        }
LABEL_62:
        ++*(_DWORD *)(a1 + 328);
        BUG();
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 312);
    }
    sub_1278800(a1 + 312, 2 * v5);
    v18 = *(_DWORD *)(a1 + 336);
    if ( v18 )
    {
      v19 = v18 - 1;
      v20 = *(_QWORD *)(a1 + 320);
      v21 = v19 & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
      v22 = *(_DWORD *)(a1 + 328) + 1;
      v14 = (__int64 *)(v20 + 16LL * v21);
      v23 = *v14;
      if ( *v14 != v3 )
      {
        v35 = 1;
        v36 = 0;
        while ( v23 != -8 )
        {
          if ( v23 == -16 && !v36 )
            v36 = v14;
          v21 = v19 & (v35 + v21);
          v14 = (__int64 *)(v20 + 16LL * v21);
          v23 = *v14;
          if ( *v14 == v3 )
            goto LABEL_24;
          ++v35;
        }
        if ( v36 )
          v14 = v36;
      }
      goto LABEL_24;
    }
    goto LABEL_62;
  }
LABEL_5:
  if ( v8 == (__int64 *)(v6 + 16LL * v5) )
    goto LABEL_10;
  return *((unsigned __int8 *)v8 + 8);
}
