// Function: sub_1361F30
// Address: 0x1361f30
//
__int64 __fastcall sub_1361F30(__int64 *a1, __int64 a2)
{
  __int64 v3; // rbx
  __int64 result; // rax
  unsigned int v5; // esi
  __int64 v6; // rdi
  __int64 v7; // rdx
  int v8; // r11d
  __int64 *v9; // r10
  unsigned __int64 v10; // rcx
  unsigned __int64 v11; // rcx
  unsigned __int64 v12; // rax
  unsigned int i; // r8d
  __int64 *v14; // r13
  __int64 v15; // rcx
  unsigned int v16; // r8d
  int v17; // ecx
  int v18; // ecx
  __int64 v19; // rdx
  int v20; // r8d
  __int64 *v21; // rdi
  unsigned __int64 v22; // rsi
  unsigned __int64 v23; // rsi
  unsigned int v24; // eax
  __int64 v25; // rsi
  unsigned int v26; // eax
  int v27; // edx
  int v28; // ecx
  int v29; // edx
  int v30; // edx
  __int64 v31; // rdi
  int v32; // r8d
  unsigned int j; // eax
  __int64 *v34; // rsi
  __int64 v35; // rcx
  unsigned int v36; // eax
  int v37; // [rsp+8h] [rbp-28h]

  v3 = *a1;
  if ( !*a1 || !*(_BYTE *)(v3 + 1) )
    return sub_135D890(a2);
  v5 = *(_DWORD *)(v3 + 32);
  v6 = v3 + 8;
  if ( !v5 )
  {
    ++*(_QWORD *)(v3 + 8);
    goto LABEL_14;
  }
  v7 = *(_QWORD *)(v3 + 16);
  v8 = 1;
  v9 = 0;
  v10 = (((((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)
         | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))
        - 1
        - ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32)) >> 22)
      ^ ((((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)
        | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))
       - 1
       - ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32));
  v11 = ((9 * (((v10 - 1 - (v10 << 13)) >> 8) ^ (v10 - 1 - (v10 << 13)))) >> 15)
      ^ (9 * (((v10 - 1 - (v10 << 13)) >> 8) ^ (v10 - 1 - (v10 << 13))));
  v12 = ((v11 - 1 - (v11 << 27)) >> 31) ^ (v11 - 1 - (v11 << 27));
  for ( i = v12 & (v5 - 1); ; i = (v5 - 1) & v16 )
  {
    v14 = (__int64 *)(v7 + 24LL * i);
    v15 = *v14;
    if ( a2 == *v14 && a2 == v14[1] )
      return *((unsigned __int8 *)v14 + 16);
    if ( v15 == -8 )
      break;
    if ( v15 == -16 && v14[1] == -16 && !v9 )
      v9 = (__int64 *)(v7 + 24LL * i);
LABEL_12:
    v16 = v8 + i;
    ++v8;
  }
  if ( v14[1] != -8 )
    goto LABEL_12;
  v27 = *(_DWORD *)(v3 + 24);
  if ( v9 )
    v14 = v9;
  ++*(_QWORD *)(v3 + 8);
  v28 = v27 + 1;
  if ( 4 * (v27 + 1) < 3 * v5 )
  {
    if ( v5 - *(_DWORD *)(v3 + 28) - v28 > v5 >> 3 )
      goto LABEL_28;
    v37 = v12;
    sub_1350E40(v6, v5);
    v29 = *(_DWORD *)(v3 + 32);
    if ( v29 )
    {
      v30 = v29 - 1;
      v14 = 0;
      v32 = 1;
      for ( j = v30 & v37; ; j = v30 & v36 )
      {
        v31 = *(_QWORD *)(v3 + 16);
        v34 = (__int64 *)(v31 + 24LL * j);
        v35 = *v34;
        if ( a2 == *v34 && a2 == v34[1] )
        {
          v14 = (__int64 *)(v31 + 24LL * j);
          v28 = *(_DWORD *)(v3 + 24) + 1;
          goto LABEL_28;
        }
        if ( v35 == -8 )
        {
          if ( v34[1] == -8 )
          {
            if ( !v14 )
              v14 = (__int64 *)(v31 + 24LL * j);
            v28 = *(_DWORD *)(v3 + 24) + 1;
            goto LABEL_28;
          }
        }
        else if ( v35 == -16 && v34[1] == -16 && !v14 )
        {
          v14 = (__int64 *)(v31 + 24LL * j);
        }
        v36 = v32 + j;
        ++v32;
      }
    }
LABEL_56:
    ++*(_DWORD *)(v3 + 24);
    BUG();
  }
LABEL_14:
  sub_1350E40(v6, 2 * v5);
  v17 = *(_DWORD *)(v3 + 32);
  if ( !v17 )
    goto LABEL_56;
  v18 = v17 - 1;
  v19 = *(_QWORD *)(v3 + 16);
  v20 = 1;
  v21 = 0;
  v22 = (((((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)
         | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))
        - 1
        - ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32)) >> 22)
      ^ ((((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)
        | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))
       - 1
       - ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32));
  v23 = ((9 * (((v22 - 1 - (v22 << 13)) >> 8) ^ (v22 - 1 - (v22 << 13)))) >> 15)
      ^ (9 * (((v22 - 1 - (v22 << 13)) >> 8) ^ (v22 - 1 - (v22 << 13))));
  v24 = v18 & (((v23 - 1 - (v23 << 27)) >> 31) ^ (v23 - 1 - ((_DWORD)v23 << 27)));
  while ( 2 )
  {
    v14 = (__int64 *)(v19 + 24LL * v24);
    v25 = *v14;
    if ( a2 == *v14 && a2 == v14[1] )
    {
      v28 = *(_DWORD *)(v3 + 24) + 1;
      goto LABEL_28;
    }
    if ( v25 != -8 )
    {
      if ( v25 == -16 && v14[1] == -16 && !v21 )
        v21 = (__int64 *)(v19 + 24LL * v24);
      goto LABEL_22;
    }
    if ( v14[1] != -8 )
    {
LABEL_22:
      v26 = v20 + v24;
      ++v20;
      v24 = v18 & v26;
      continue;
    }
    break;
  }
  if ( v21 )
    v14 = v21;
  v28 = *(_DWORD *)(v3 + 24) + 1;
LABEL_28:
  *(_DWORD *)(v3 + 24) = v28;
  if ( *v14 != -8 || v14[1] != -8 )
    --*(_DWORD *)(v3 + 28);
  *v14 = a2;
  v14[1] = a2;
  *((_BYTE *)v14 + 16) = 0;
  result = sub_135D890(a2);
  *((_BYTE *)v14 + 16) = result;
  return result;
}
