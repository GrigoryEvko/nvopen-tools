// Function: sub_1645D80
// Address: 0x1645d80
//
__int64 *__fastcall sub_1645D80(__int64 *a1, __int64 a2)
{
  __int64 v4; // rbx
  unsigned int v5; // esi
  __int64 v6; // rdi
  int v7; // r10d
  __int64 v8; // rdx
  __int64 **v9; // r11
  unsigned __int64 v10; // rcx
  unsigned __int64 v11; // rcx
  unsigned __int64 v12; // rax
  unsigned int i; // r8d
  __int64 **v14; // r14
  __int64 *v15; // rcx
  unsigned int v16; // r8d
  __int64 *result; // rax
  int v18; // edx
  int v19; // ecx
  int v20; // ecx
  int v21; // ecx
  __int64 v22; // rdx
  int v23; // r8d
  __int64 **v24; // rdi
  unsigned __int64 v25; // rsi
  unsigned __int64 v26; // rsi
  unsigned int j; // eax
  __int64 *v28; // rsi
  unsigned int v29; // eax
  int v30; // edx
  int v31; // edx
  __int64 v32; // rsi
  int v33; // r8d
  unsigned int k; // eax
  __int64 *v35; // rcx
  unsigned int v36; // eax
  __int64 *v37; // [rsp+8h] [rbp-28h]
  int v38; // [rsp+8h] [rbp-28h]

  v4 = *(_QWORD *)*a1;
  v5 = *(_DWORD *)(v4 + 2536);
  v6 = v4 + 2512;
  if ( !v5 )
  {
    ++*(_QWORD *)(v4 + 2512);
    goto LABEL_25;
  }
  v7 = 1;
  v8 = *(_QWORD *)(v4 + 2520);
  v9 = 0;
  v10 = ((((unsigned int)(37 * a2) | ((unsigned __int64)(((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4)) << 32))
        - 1
        - ((unsigned __int64)(unsigned int)(37 * a2) << 32)) >> 22)
      ^ (((unsigned int)(37 * a2) | ((unsigned __int64)(((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4)) << 32))
       - 1
       - ((unsigned __int64)(unsigned int)(37 * a2) << 32));
  v11 = ((9 * (((v10 - 1 - (v10 << 13)) >> 8) ^ (v10 - 1 - (v10 << 13)))) >> 15)
      ^ (9 * (((v10 - 1 - (v10 << 13)) >> 8) ^ (v10 - 1 - (v10 << 13))));
  v12 = ((v11 - 1 - (v11 << 27)) >> 31) ^ (v11 - 1 - (v11 << 27));
  for ( i = v12 & (v5 - 1); ; i = (v5 - 1) & v16 )
  {
    v14 = (__int64 **)(v8 + 24LL * i);
    v15 = *v14;
    if ( a1 == *v14 && (__int64 *)a2 == v14[1] )
    {
      result = v14[2];
      if ( !result )
        goto LABEL_21;
      return result;
    }
    if ( v15 == (__int64 *)-8LL )
      break;
    if ( v15 == (__int64 *)-16LL && v14[1] == (__int64 *)-2LL && !v9 )
      v9 = (__int64 **)(v8 + 24LL * i);
LABEL_9:
    v16 = v7 + i;
    ++v7;
  }
  if ( v14[1] != (__int64 *)-1LL )
    goto LABEL_9;
  v18 = *(_DWORD *)(v4 + 2528);
  if ( v9 )
    v14 = v9;
  ++*(_QWORD *)(v4 + 2512);
  v19 = v18 + 1;
  if ( 4 * (v18 + 1) >= 3 * v5 )
  {
LABEL_25:
    sub_1645AD0(v6, 2 * v5);
    v20 = *(_DWORD *)(v4 + 2536);
    if ( v20 )
    {
      v21 = v20 - 1;
      v23 = 1;
      v24 = 0;
      v25 = ((((unsigned int)(37 * a2) | ((unsigned __int64)(((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4)) << 32))
            - 1
            - ((unsigned __int64)(unsigned int)(37 * a2) << 32)) >> 22)
          ^ (((unsigned int)(37 * a2) | ((unsigned __int64)(((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4)) << 32))
           - 1
           - ((unsigned __int64)(unsigned int)(37 * a2) << 32));
      v26 = ((9 * (((v25 - 1 - (v25 << 13)) >> 8) ^ (v25 - 1 - (v25 << 13)))) >> 15)
          ^ (9 * (((v25 - 1 - (v25 << 13)) >> 8) ^ (v25 - 1 - (v25 << 13))));
      for ( j = v21 & (((v26 - 1 - (v26 << 27)) >> 31) ^ (v26 - 1 - ((_DWORD)v26 << 27))); ; j = v21 & v29 )
      {
        v22 = *(_QWORD *)(v4 + 2520);
        v14 = (__int64 **)(v22 + 24LL * j);
        v28 = *v14;
        if ( a1 == *v14 && (__int64 *)a2 == v14[1] )
          break;
        if ( v28 == (__int64 *)-8LL )
        {
          if ( v14[1] == (__int64 *)-1LL )
          {
LABEL_48:
            if ( v24 )
              v14 = v24;
            v19 = *(_DWORD *)(v4 + 2528) + 1;
            goto LABEL_18;
          }
        }
        else if ( v28 == (__int64 *)-16LL && v14[1] == (__int64 *)-2LL && !v24 )
        {
          v24 = (__int64 **)(v22 + 24LL * j);
        }
        v29 = v23 + j;
        ++v23;
      }
      goto LABEL_44;
    }
LABEL_53:
    ++*(_DWORD *)(v4 + 2528);
    BUG();
  }
  if ( v5 - *(_DWORD *)(v4 + 2532) - v19 <= v5 >> 3 )
  {
    v38 = v12;
    sub_1645AD0(v6, v5);
    v30 = *(_DWORD *)(v4 + 2536);
    if ( v30 )
    {
      v31 = v30 - 1;
      v24 = 0;
      v33 = 1;
      for ( k = v31 & v38; ; k = v31 & v36 )
      {
        v32 = *(_QWORD *)(v4 + 2520);
        v14 = (__int64 **)(v32 + 24LL * k);
        v35 = *v14;
        if ( a1 == *v14 && (__int64 *)a2 == v14[1] )
          break;
        if ( v35 == (__int64 *)-8LL )
        {
          if ( v14[1] == (__int64 *)-1LL )
            goto LABEL_48;
        }
        else if ( v35 == (__int64 *)-16LL && v14[1] == (__int64 *)-2LL && !v24 )
        {
          v24 = (__int64 **)(v32 + 24LL * k);
        }
        v36 = v33 + k;
        ++v33;
      }
LABEL_44:
      v19 = *(_DWORD *)(v4 + 2528) + 1;
      goto LABEL_18;
    }
    goto LABEL_53;
  }
LABEL_18:
  *(_DWORD *)(v4 + 2528) = v19;
  if ( *v14 != (__int64 *)-8LL || v14[1] != (__int64 *)-1LL )
    --*(_DWORD *)(v4 + 2532);
  *v14 = a1;
  v14[1] = (__int64 *)a2;
  v14[2] = 0;
LABEL_21:
  v37 = (__int64 *)sub_145CBF0((__int64 *)(v4 + 2272), 40, 16);
  sub_1643E90(v37, a1, a2);
  v14[2] = v37;
  return v37;
}
