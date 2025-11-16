// Function: sub_1FE5190
// Address: 0x1fe5190
//
unsigned __int64 __fastcall sub_1FE5190(__int64 a1, __int64 a2, __int64 a3, int a4)
{
  __int64 v8; // rdi
  unsigned int v9; // esi
  __int64 v10; // rdx
  int v11; // r10d
  __int64 *v12; // r11
  unsigned __int64 v13; // rcx
  unsigned __int64 v14; // rcx
  unsigned __int64 result; // rax
  unsigned int j; // r8d
  __int64 *v17; // rcx
  __int64 v18; // r15
  unsigned int v19; // r8d
  int v20; // edx
  int v21; // r8d
  int v22; // ecx
  __int64 v23; // rdx
  int v24; // r8d
  __int64 *v25; // r9
  int v26; // edi
  unsigned __int64 v27; // rsi
  unsigned __int64 v28; // rsi
  unsigned int i; // eax
  __int64 v30; // rsi
  unsigned int v31; // eax
  int v32; // edx
  int v33; // edx
  __int64 v34; // rdi
  int v35; // r8d
  unsigned int k; // eax
  __int64 v37; // rsi
  unsigned int v38; // eax
  int v39; // [rsp+8h] [rbp-38h]

  v8 = a1 + 80;
  v9 = *(_DWORD *)(a1 + 104);
  if ( !v9 )
  {
    ++*(_QWORD *)(a1 + 80);
LABEL_23:
    sub_1FE4EE0(v8, 2 * v9);
    v22 = *(_DWORD *)(a1 + 104);
    if ( v22 )
    {
      v24 = 1;
      v25 = 0;
      v26 = v22 - 1;
      v27 = (((((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
             | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))
            - 1
            - ((unsigned __int64)(((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)) << 32)) >> 22)
          ^ ((((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
            | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))
           - 1
           - ((unsigned __int64)(((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)) << 32));
      v28 = ((9 * (((v27 - 1 - (v27 << 13)) >> 8) ^ (v27 - 1 - (v27 << 13)))) >> 15)
          ^ (9 * (((v27 - 1 - (v27 << 13)) >> 8) ^ (v27 - 1 - (v27 << 13))));
      for ( i = (v22 - 1) & (((v28 - 1 - (v28 << 27)) >> 31) ^ (v28 - 1 - ((_DWORD)v28 << 27))); ; i = v26 & v31 )
      {
        v23 = *(_QWORD *)(a1 + 88);
        v17 = (__int64 *)(v23 + 24LL * i);
        v30 = *v17;
        if ( a2 == *v17 && a3 == v17[1] )
          break;
        if ( v30 == -8 )
        {
          if ( v17[1] == -8 )
          {
LABEL_46:
            result = *(unsigned int *)(a1 + 96);
            if ( v25 )
              v17 = v25;
            v21 = result + 1;
            goto LABEL_17;
          }
        }
        else if ( v30 == -16 && v17[1] == -16 && !v25 )
        {
          v25 = (__int64 *)(v23 + 24LL * i);
        }
        v31 = v24 + i;
        ++v24;
      }
      goto LABEL_42;
    }
LABEL_51:
    ++*(_DWORD *)(a1 + 96);
    BUG();
  }
  v10 = *(_QWORD *)(a1 + 88);
  v11 = 1;
  v12 = 0;
  v13 = (((((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
         | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))
        - 1
        - ((unsigned __int64)(((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)) << 32)) >> 22)
      ^ ((((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
        | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))
       - 1
       - ((unsigned __int64)(((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)) << 32));
  v14 = ((9 * (((v13 - 1 - (v13 << 13)) >> 8) ^ (v13 - 1 - (v13 << 13)))) >> 15)
      ^ (9 * (((v13 - 1 - (v13 << 13)) >> 8) ^ (v13 - 1 - (v13 << 13))));
  result = ((v14 - 1 - (v14 << 27)) >> 31) ^ (v14 - 1 - (v14 << 27));
  for ( j = result & (v9 - 1); ; j = (v9 - 1) & v19 )
  {
    v17 = (__int64 *)(v10 + 24LL * j);
    v18 = *v17;
    if ( a2 == *v17 && a3 == v17[1] )
      goto LABEL_11;
    if ( v18 == -8 )
      break;
    if ( v18 == -16 && v17[1] == -16 && !v12 )
      v12 = (__int64 *)(v10 + 24LL * j);
LABEL_9:
    v19 = v11 + j;
    ++v11;
  }
  if ( v17[1] != -8 )
    goto LABEL_9;
  v20 = *(_DWORD *)(a1 + 96);
  if ( v12 )
    v17 = v12;
  ++*(_QWORD *)(a1 + 80);
  v21 = v20 + 1;
  if ( 4 * (v20 + 1) >= 3 * v9 )
    goto LABEL_23;
  if ( v9 - *(_DWORD *)(a1 + 100) - v21 > v9 >> 3 )
    goto LABEL_17;
  v39 = result;
  sub_1FE4EE0(v8, v9);
  v32 = *(_DWORD *)(a1 + 104);
  if ( !v32 )
    goto LABEL_51;
  v33 = v32 - 1;
  v25 = 0;
  v35 = 1;
  for ( k = v33 & v39; ; k = v33 & v38 )
  {
    v34 = *(_QWORD *)(a1 + 88);
    v17 = (__int64 *)(v34 + 24LL * k);
    v37 = *v17;
    if ( a2 == *v17 && a3 == v17[1] )
      break;
    if ( v37 == -8 )
    {
      if ( v17[1] == -8 )
        goto LABEL_46;
    }
    else if ( v37 == -16 && v17[1] == -16 && !v25 )
    {
      v25 = (__int64 *)(v34 + 24LL * k);
    }
    v38 = v35 + k;
    ++v35;
  }
LABEL_42:
  result = *(unsigned int *)(a1 + 96);
  v21 = result + 1;
LABEL_17:
  *(_DWORD *)(a1 + 96) = v21;
  if ( *v17 != -8 || v17[1] != -8 )
    --*(_DWORD *)(a1 + 100);
  *v17 = a2;
  v17[1] = a3;
  *((_DWORD *)v17 + 4) = 0;
LABEL_11:
  *((_DWORD *)v17 + 4) = a4;
  return result;
}
