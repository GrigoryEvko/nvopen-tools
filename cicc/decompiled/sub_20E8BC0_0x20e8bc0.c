// Function: sub_20E8BC0
// Address: 0x20e8bc0
//
__int64 __fastcall sub_20E8BC0(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  unsigned int v4; // r13d
  __int64 v5; // r12
  unsigned int v6; // esi
  __int64 v7; // rdi
  __int64 v8; // rdx
  int v9; // r14d
  unsigned __int64 v10; // rcx
  unsigned __int64 v11; // rcx
  unsigned __int64 v12; // rax
  unsigned int *v13; // rcx
  unsigned int i; // r10d
  int *v15; // r8
  int v16; // r9d
  unsigned int v17; // r10d
  __int64 result; // rax
  int v19; // edx
  int v20; // r8d
  int v21; // ecx
  __int64 v22; // rdx
  unsigned int *v23; // r9
  int v24; // r8d
  int v25; // edi
  unsigned __int64 v26; // rsi
  unsigned __int64 v27; // rsi
  unsigned int j; // eax
  unsigned int v29; // esi
  unsigned int v30; // eax
  int v31; // edx
  int v32; // edx
  __int64 v33; // rdi
  int v34; // r8d
  unsigned int k; // eax
  unsigned int v36; // esi
  unsigned int v37; // eax
  int v38; // [rsp+8h] [rbp-28h]

  v3 = sub_20E8320(a1, *(unsigned __int16 *)(a2 + 6));
  v4 = *(_DWORD *)(a1 + 8);
  v5 = v3;
  sub_20E8610(a1, v4);
  v6 = *(_DWORD *)(a1 + 56);
  v7 = a1 + 32;
  if ( !v6 )
  {
    ++*(_QWORD *)(a1 + 32);
    goto LABEL_23;
  }
  v8 = *(_QWORD *)(a1 + 40);
  v9 = 1;
  v10 = ((((unsigned int)(37 * v5) | ((unsigned __int64)(37 * v4) << 32))
        - 1
        - ((unsigned __int64)(unsigned int)(37 * v5) << 32)) >> 22)
      ^ (((unsigned int)(37 * v5) | ((unsigned __int64)(37 * v4) << 32))
       - 1
       - ((unsigned __int64)(unsigned int)(37 * v5) << 32));
  v11 = ((9 * (((v10 - 1 - (v10 << 13)) >> 8) ^ (v10 - 1 - (v10 << 13)))) >> 15)
      ^ (9 * (((v10 - 1 - (v10 << 13)) >> 8) ^ (v10 - 1 - (v10 << 13))));
  v12 = ((v11 - 1 - (v11 << 27)) >> 31) ^ (v11 - 1 - (v11 << 27));
  v13 = 0;
  for ( i = v12 & (v6 - 1); ; i = (v6 - 1) & v17 )
  {
    v15 = (int *)(v8 + 24LL * i);
    v16 = *v15;
    if ( v4 == *v15 && v5 == *((_QWORD *)v15 + 1) )
    {
      result = (unsigned int)v15[4];
      *(_DWORD *)(a1 + 8) = result;
      return result;
    }
    if ( v16 == -1 )
      break;
    if ( v16 == -2 && *((_QWORD *)v15 + 1) == -2 && !v13 )
      v13 = (unsigned int *)(v8 + 24LL * i);
LABEL_9:
    v17 = v9 + i;
    ++v9;
  }
  if ( *((_QWORD *)v15 + 1) != -1 )
    goto LABEL_9;
  v19 = *(_DWORD *)(a1 + 48);
  if ( !v13 )
    v13 = (unsigned int *)v15;
  ++*(_QWORD *)(a1 + 32);
  v20 = v19 + 1;
  if ( 4 * (v19 + 1) >= 3 * v6 )
  {
LABEL_23:
    sub_20E8370(v7, 2 * v6);
    v21 = *(_DWORD *)(a1 + 56);
    if ( v21 )
    {
      v23 = 0;
      v24 = 1;
      v25 = v21 - 1;
      v26 = ((((unsigned int)(37 * v5) | ((unsigned __int64)(37 * v4) << 32))
            - 1
            - ((unsigned __int64)(unsigned int)(37 * v5) << 32)) >> 22)
          ^ (((unsigned int)(37 * v5) | ((unsigned __int64)(37 * v4) << 32))
           - 1
           - ((unsigned __int64)(unsigned int)(37 * v5) << 32));
      v27 = ((9 * (((v26 - 1 - (v26 << 13)) >> 8) ^ (v26 - 1 - (v26 << 13)))) >> 15)
          ^ (9 * (((v26 - 1 - (v26 << 13)) >> 8) ^ (v26 - 1 - (v26 << 13))));
      for ( j = (v21 - 1) & (((v27 - 1 - (v27 << 27)) >> 31) ^ (v27 - 1 - ((_DWORD)v27 << 27))); ; j = v25 & v30 )
      {
        v22 = *(_QWORD *)(a1 + 40);
        v13 = (unsigned int *)(v22 + 24LL * j);
        v29 = *v13;
        if ( v4 == *v13 && v5 == *((_QWORD *)v13 + 1) )
          break;
        if ( v29 == -1 )
        {
          if ( *((_QWORD *)v13 + 1) == -1 )
          {
LABEL_46:
            if ( v23 )
              v13 = v23;
            v20 = *(_DWORD *)(a1 + 48) + 1;
            goto LABEL_17;
          }
        }
        else if ( v29 == -2 && *((_QWORD *)v13 + 1) == -2 && !v23 )
        {
          v23 = (unsigned int *)(v22 + 24LL * j);
        }
        v30 = v24 + j;
        ++v24;
      }
      goto LABEL_42;
    }
LABEL_51:
    ++*(_DWORD *)(a1 + 48);
    BUG();
  }
  if ( v6 - *(_DWORD *)(a1 + 52) - v20 <= v6 >> 3 )
  {
    v38 = v12;
    sub_20E8370(v7, v6);
    v31 = *(_DWORD *)(a1 + 56);
    if ( v31 )
    {
      v32 = v31 - 1;
      v23 = 0;
      v34 = 1;
      for ( k = v32 & v38; ; k = v32 & v37 )
      {
        v33 = *(_QWORD *)(a1 + 40);
        v13 = (unsigned int *)(v33 + 24LL * k);
        v36 = *v13;
        if ( v4 == *v13 && v5 == *((_QWORD *)v13 + 1) )
          break;
        if ( v36 == -1 )
        {
          if ( *((_QWORD *)v13 + 1) == -1 )
            goto LABEL_46;
        }
        else if ( v36 == -2 && *((_QWORD *)v13 + 1) == -2 && !v23 )
        {
          v23 = (unsigned int *)(v33 + 24LL * k);
        }
        v37 = v34 + k;
        ++v34;
      }
LABEL_42:
      v20 = *(_DWORD *)(a1 + 48) + 1;
      goto LABEL_17;
    }
    goto LABEL_51;
  }
LABEL_17:
  *(_DWORD *)(a1 + 48) = v20;
  if ( *v13 != -1 || *((_QWORD *)v13 + 1) != -1 )
    --*(_DWORD *)(a1 + 52);
  *v13 = v4;
  *((_QWORD *)v13 + 1) = v5;
  v13[4] = 0;
  *(_DWORD *)(a1 + 8) = 0;
  return 0;
}
