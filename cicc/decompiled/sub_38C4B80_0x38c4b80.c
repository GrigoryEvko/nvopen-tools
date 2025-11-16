// Function: sub_38C4B80
// Address: 0x38c4b80
//
__int64 __fastcall sub_38C4B80(__int64 a1, int a2, int a3)
{
  __int64 v5; // rdi
  unsigned int v7; // esi
  int v8; // r10d
  int *v9; // r9
  __int64 v10; // rdx
  unsigned __int64 v11; // rcx
  unsigned __int64 v12; // rcx
  unsigned __int64 v13; // rax
  unsigned int i; // ecx
  int *v15; // r14
  int v16; // r11d
  unsigned int v17; // ecx
  __int64 result; // rax
  int v19; // edx
  int v20; // ecx
  int v21; // ecx
  int v22; // ecx
  int *v23; // rdi
  __int64 v24; // rdx
  int v25; // r8d
  unsigned __int64 v26; // rsi
  unsigned __int64 v27; // rsi
  unsigned int j; // eax
  int v29; // esi
  unsigned int v30; // eax
  int v31; // edx
  int v32; // edx
  int v33; // r8d
  __int64 v34; // rsi
  unsigned int k; // eax
  int v36; // ecx
  unsigned int v37; // eax
  int v38; // [rsp+8h] [rbp-28h]

  v5 = a1 + 600;
  v7 = *(_DWORD *)(a1 + 624);
  if ( !v7 )
  {
    ++*(_QWORD *)(a1 + 600);
    goto LABEL_25;
  }
  v8 = 1;
  v9 = 0;
  v10 = *(_QWORD *)(a1 + 608);
  v11 = ((((unsigned int)(37 * a3) | ((unsigned __int64)(unsigned int)(37 * a2) << 32))
        - 1
        - ((unsigned __int64)(unsigned int)(37 * a3) << 32)) >> 22)
      ^ (((unsigned int)(37 * a3) | ((unsigned __int64)(unsigned int)(37 * a2) << 32))
       - 1
       - ((unsigned __int64)(unsigned int)(37 * a3) << 32));
  v12 = ((9 * (((v11 - 1 - (v11 << 13)) >> 8) ^ (v11 - 1 - (v11 << 13)))) >> 15)
      ^ (9 * (((v11 - 1 - (v11 << 13)) >> 8) ^ (v11 - 1 - (v11 << 13))));
  v13 = ((v12 - 1 - (v12 << 27)) >> 31) ^ (v12 - 1 - (v12 << 27));
  for ( i = v13 & (v7 - 1); ; i = (v7 - 1) & v17 )
  {
    v15 = (int *)(v10 + 16LL * i);
    v16 = *v15;
    if ( a2 == *v15 && a3 == v15[1] )
    {
      result = *((_QWORD *)v15 + 1);
      if ( !result )
        goto LABEL_21;
      return result;
    }
    if ( v16 == -1 )
      break;
    if ( v16 == -2 && v15[1] == -2 && !v9 )
      v9 = (int *)(v10 + 16LL * i);
LABEL_9:
    v17 = v8 + i;
    ++v8;
  }
  if ( v15[1] != -1 )
    goto LABEL_9;
  v19 = *(_DWORD *)(a1 + 616);
  if ( v9 )
    v15 = v9;
  ++*(_QWORD *)(a1 + 600);
  v20 = v19 + 1;
  if ( 4 * (v19 + 1) >= 3 * v7 )
  {
LABEL_25:
    sub_38C48F0(v5, 2 * v7);
    v21 = *(_DWORD *)(a1 + 624);
    if ( v21 )
    {
      v22 = v21 - 1;
      v23 = 0;
      v24 = *(_QWORD *)(a1 + 608);
      v25 = 1;
      v26 = ((((unsigned int)(37 * a3) | ((unsigned __int64)(unsigned int)(37 * a2) << 32))
            - 1
            - ((unsigned __int64)(unsigned int)(37 * a3) << 32)) >> 22)
          ^ (((unsigned int)(37 * a3) | ((unsigned __int64)(unsigned int)(37 * a2) << 32))
           - 1
           - ((unsigned __int64)(unsigned int)(37 * a3) << 32));
      v27 = ((9 * (((v26 - 1 - (v26 << 13)) >> 8) ^ (v26 - 1 - (v26 << 13)))) >> 15)
          ^ (9 * (((v26 - 1 - (v26 << 13)) >> 8) ^ (v26 - 1 - (v26 << 13))));
      for ( j = v22 & (((v27 - 1 - (v27 << 27)) >> 31) ^ (v27 - 1 - ((_DWORD)v27 << 27))); ; j = v22 & v30 )
      {
        v15 = (int *)(v24 + 16LL * j);
        v29 = *v15;
        if ( a2 == *v15 && a3 == v15[1] )
          break;
        if ( v29 == -1 )
        {
          if ( v15[1] == -1 )
          {
LABEL_48:
            if ( v23 )
              v15 = v23;
            v20 = *(_DWORD *)(a1 + 616) + 1;
            goto LABEL_18;
          }
        }
        else if ( v29 == -2 && v15[1] == -2 && !v23 )
        {
          v23 = (int *)(v24 + 16LL * j);
        }
        v30 = v25 + j;
        ++v25;
      }
      goto LABEL_44;
    }
LABEL_53:
    ++*(_DWORD *)(a1 + 616);
    BUG();
  }
  if ( v7 - *(_DWORD *)(a1 + 620) - v20 <= v7 >> 3 )
  {
    v38 = v13;
    sub_38C48F0(v5, v7);
    v31 = *(_DWORD *)(a1 + 624);
    if ( v31 )
    {
      v32 = v31 - 1;
      v33 = 1;
      v23 = 0;
      for ( k = v32 & v38; ; k = v32 & v37 )
      {
        v34 = *(_QWORD *)(a1 + 608);
        v15 = (int *)(v34 + 16LL * k);
        v36 = *v15;
        if ( a2 == *v15 && a3 == v15[1] )
          break;
        if ( v36 == -1 )
        {
          if ( v15[1] == -1 )
            goto LABEL_48;
        }
        else if ( v36 == -2 && v15[1] == -2 && !v23 )
        {
          v23 = (int *)(v34 + 16LL * k);
        }
        v37 = v33 + k;
        ++v33;
      }
LABEL_44:
      v20 = *(_DWORD *)(a1 + 616) + 1;
      goto LABEL_18;
    }
    goto LABEL_53;
  }
LABEL_18:
  *(_DWORD *)(a1 + 616) = v20;
  if ( *v15 != -1 || v15[1] != -1 )
    --*(_DWORD *)(a1 + 620);
  *v15 = a2;
  v15[1] = a3;
  *((_QWORD *)v15 + 1) = 0;
LABEL_21:
  result = sub_38BFA60(a1, 0);
  *((_QWORD *)v15 + 1) = result;
  return result;
}
