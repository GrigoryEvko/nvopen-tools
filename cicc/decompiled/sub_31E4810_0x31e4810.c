// Function: sub_31E4810
// Address: 0x31e4810
//
__int64 __fastcall sub_31E4810(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  int v4; // r13d
  int v5; // r14d
  unsigned int v6; // esi
  __int64 v7; // rdx
  int v8; // r10d
  __int64 v9; // r9
  unsigned int j; // ecx
  __int64 v11; // rbx
  unsigned int v12; // ecx
  int v14; // r11d
  int v15; // ecx
  int v16; // ecx
  __int64 v17; // rdi
  __int64 v18; // rdx
  __int64 v19; // r8
  unsigned int i; // eax
  int v21; // eax
  int v22; // edx
  __int64 v23; // rcx
  int v24; // edx
  __int64 v25; // rsi
  unsigned int k; // eax
  int v27; // eax
  int v28; // ecx
  int v29; // esi
  char *v30; // [rsp+10h] [rbp-50h] BYREF
  char v31; // [rsp+30h] [rbp-30h]
  char v32; // [rsp+31h] [rbp-2Fh]

  v3 = a1 + 408;
  v4 = *(_DWORD *)(a2 + 252);
  v5 = *(_DWORD *)(a2 + 256);
  v6 = *(_DWORD *)(a1 + 432);
  if ( !v6 )
  {
    ++*(_QWORD *)(a1 + 408);
LABEL_14:
    sub_31E45C0(v3, 2 * v6);
    v15 = *(_DWORD *)(a1 + 432);
    if ( v15 )
    {
      v16 = v15 - 1;
      v17 = 0;
      v18 = *(_QWORD *)(a1 + 416);
      v19 = 1;
      for ( i = v16
              & (((0xBF58476D1CE4E5B9LL * ((unsigned int)(37 * v5) | ((unsigned __int64)(unsigned int)(37 * v4) << 32))) >> 31)
               ^ (756364221 * v5)); ; i = v16 & v21 )
      {
        v11 = v18 + 16LL * i;
        if ( v4 == *(_DWORD *)v11 && v5 == *(_DWORD *)(v11 + 4) )
          break;
        if ( !*(_DWORD *)v11 )
        {
          v29 = *(_DWORD *)(v11 + 4);
          if ( v29 == -1 )
          {
LABEL_47:
            if ( v17 )
              v11 = v17;
            v23 = (unsigned int)(*(_DWORD *)(a1 + 424) + 1);
            goto LABEL_23;
          }
          if ( v29 == -2 && !v17 )
            v17 = v18 + 16LL * i;
        }
        v21 = v19 + i;
        v19 = (unsigned int)(v19 + 1);
      }
      goto LABEL_34;
    }
LABEL_50:
    ++*(_DWORD *)(a1 + 424);
    BUG();
  }
  v7 = *(_QWORD *)(a1 + 416);
  v8 = 1;
  v9 = 0;
  for ( j = (((0xBF58476D1CE4E5B9LL * ((unsigned int)(37 * v5) | ((unsigned __int64)(unsigned int)(37 * v4) << 32))) >> 31)
           ^ (756364221 * v5))
          & (v6 - 1); ; j = (v6 - 1) & v12 )
  {
    v11 = v7 + 16LL * j;
    if ( v4 == *(_DWORD *)v11 && v5 == *(_DWORD *)(v11 + 4) )
      return *(_QWORD *)(v11 + 8);
    if ( !*(_DWORD *)v11 )
      break;
LABEL_5:
    v12 = v8 + j;
    ++v8;
  }
  v14 = *(_DWORD *)(v11 + 4);
  if ( v14 != -1 )
  {
    if ( v14 == -2 && !v9 )
      v9 = v7 + 16LL * j;
    goto LABEL_5;
  }
  v22 = *(_DWORD *)(a1 + 424);
  if ( v9 )
    v11 = v9;
  ++*(_QWORD *)(a1 + 408);
  v23 = (unsigned int)(v22 + 1);
  if ( 4 * (int)v23 >= 3 * v6 )
    goto LABEL_14;
  v18 = v6 - *(_DWORD *)(a1 + 428) - (unsigned int)v23;
  v19 = v6 >> 3;
  if ( (unsigned int)v18 > (unsigned int)v19 )
    goto LABEL_23;
  sub_31E45C0(v3, v6);
  v24 = *(_DWORD *)(a1 + 432);
  if ( !v24 )
    goto LABEL_50;
  v18 = (unsigned int)(v24 - 1);
  v19 = 1;
  v17 = 0;
  for ( k = v18
          & (((0xBF58476D1CE4E5B9LL * ((unsigned int)(37 * v5) | ((unsigned __int64)(unsigned int)(37 * v4) << 32))) >> 31)
           ^ (756364221 * v5)); ; k = v18 & v27 )
  {
    v25 = *(_QWORD *)(a1 + 416);
    v11 = v25 + 16LL * k;
    if ( v4 == *(_DWORD *)v11 && v5 == *(_DWORD *)(v11 + 4) )
      break;
    if ( !*(_DWORD *)v11 )
    {
      v28 = *(_DWORD *)(v11 + 4);
      if ( v28 == -1 )
        goto LABEL_47;
      if ( v28 == -2 && !v17 )
        v17 = v25 + 16LL * k;
    }
    v27 = v19 + k;
    v19 = (unsigned int)(v19 + 1);
  }
LABEL_34:
  v23 = (unsigned int)(*(_DWORD *)(a1 + 424) + 1);
LABEL_23:
  *(_DWORD *)(a1 + 424) = v23;
  if ( *(_DWORD *)v11 || *(_DWORD *)(v11 + 4) != -1 )
    --*(_DWORD *)(a1 + 428);
  *(_DWORD *)v11 = v4;
  *(_DWORD *)(v11 + 4) = v5;
  *(_QWORD *)(v11 + 8) = 0;
  v30 = "exception";
  v32 = 1;
  v31 = 3;
  *(_QWORD *)(v11 + 8) = sub_31DCC50(a1, (__int64 *)&v30, v18, v23, v19);
  return *(_QWORD *)(v11 + 8);
}
