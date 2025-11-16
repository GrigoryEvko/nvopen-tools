// Function: sub_25784B0
// Address: 0x25784b0
//
__int64 __fastcall sub_25784B0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v8; // eax
  __int64 v9; // rcx
  _QWORD *v10; // rax
  __int64 v11; // rbx
  __int64 *v12; // rsi
  __int64 v13; // rdi
  _QWORD *v14; // rdx
  __int64 v16; // rax
  unsigned int v17; // esi
  __int64 v18; // r9
  __int64 *v19; // r11
  int v20; // r13d
  unsigned int v21; // edx
  __int64 *v22; // rdi
  __int64 v23; // r8
  __int64 *v24; // rbx
  __int64 *v25; // r14
  __int64 v26; // r8
  unsigned int v27; // eax
  __int64 *v28; // rdi
  __int64 v29; // rcx
  unsigned int v30; // esi
  __int64 *v31; // r10
  int v32; // edx
  int v33; // eax
  __int64 v34; // rax
  __int64 v35; // rbx
  int v36; // r11d
  int v37; // eax
  _QWORD v38[7]; // [rsp+8h] [rbp-38h] BYREF

  v8 = *(_DWORD *)(a1 + 16);
  if ( v8 )
  {
    v17 = *(_DWORD *)(a1 + 24);
    if ( v17 )
    {
      v18 = *(_QWORD *)(a1 + 8);
      v19 = 0;
      v20 = 1;
      v21 = (v17 - 1) & (((unsigned int)*a2 >> 9) ^ ((unsigned int)*a2 >> 4));
      v22 = (__int64 *)(v18 + 8LL * v21);
      v23 = *v22;
      if ( *v22 == *a2 )
      {
LABEL_21:
        LODWORD(a5) = 0;
        return (unsigned int)a5;
      }
      while ( v23 != -4096 )
      {
        if ( v19 || v23 != -8192 )
          v22 = v19;
        v21 = (v17 - 1) & (v20 + v21);
        v23 = *(_QWORD *)(v18 + 8LL * v21);
        if ( *a2 == v23 )
          goto LABEL_21;
        ++v20;
        v19 = v22;
        v22 = (__int64 *)(v18 + 8LL * v21);
      }
      if ( !v19 )
        v19 = v22;
      v33 = v8 + 1;
      ++*(_QWORD *)a1;
      v38[0] = v19;
      if ( 4 * v33 < 3 * v17 )
      {
        if ( v17 - *(_DWORD *)(a1 + 20) - v33 > v17 >> 3 )
        {
LABEL_53:
          *(_DWORD *)(a1 + 16) = v33;
          if ( *v19 != -4096 )
            --*(_DWORD *)(a1 + 20);
          *v19 = *a2;
          v34 = *(unsigned int *)(a1 + 40);
          v35 = *a2;
          if ( v34 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
          {
            sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), v34 + 1, 8u, v23, v18);
            v34 = *(unsigned int *)(a1 + 40);
          }
          LODWORD(a5) = 1;
          *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8 * v34) = v35;
          ++*(_DWORD *)(a1 + 40);
          return (unsigned int)a5;
        }
LABEL_67:
        sub_CE2A30(a1, v17);
        sub_DA5B20(a1, a2, v38);
        v19 = (__int64 *)v38[0];
        v33 = *(_DWORD *)(a1 + 16) + 1;
        goto LABEL_53;
      }
    }
    else
    {
      ++*(_QWORD *)a1;
      v38[0] = 0;
    }
    v17 *= 2;
    goto LABEL_67;
  }
  v9 = *(unsigned int *)(a1 + 40);
  v10 = *(_QWORD **)(a1 + 32);
  v11 = *a2;
  v12 = &v10[v9];
  v13 = (8 * v9) >> 3;
  if ( !((8 * v9) >> 5) )
    goto LABEL_12;
  v14 = &v10[4 * ((8 * v9) >> 5)];
  do
  {
    if ( *v10 == v11 )
      goto LABEL_9;
    if ( v10[1] == v11 )
    {
      a5 = 0;
      if ( v12 == v10 + 1 )
        goto LABEL_15;
      return (unsigned int)a5;
    }
    if ( v10[2] == v11 )
    {
      a5 = 0;
      if ( v12 == v10 + 2 )
        goto LABEL_15;
      return (unsigned int)a5;
    }
    if ( v10[3] == v11 )
    {
      a5 = 0;
      if ( v12 == v10 + 3 )
        goto LABEL_15;
      return (unsigned int)a5;
    }
    v10 += 4;
  }
  while ( v14 != v10 );
  v13 = v12 - v10;
LABEL_12:
  if ( v13 != 2 )
  {
    if ( v13 != 3 )
    {
      if ( v13 != 1 )
        goto LABEL_15;
LABEL_36:
      if ( *v10 == v11 )
        goto LABEL_9;
LABEL_15:
      if ( v9 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
      {
        sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), v9 + 1, 8u, a5, a6);
        v12 = (__int64 *)(*(_QWORD *)(a1 + 32) + 8LL * *(unsigned int *)(a1 + 40));
      }
      *v12 = v11;
      v16 = (unsigned int)(*(_DWORD *)(a1 + 40) + 1);
      *(_DWORD *)(a1 + 40) = v16;
      if ( (unsigned int)v16 <= 8 )
      {
LABEL_18:
        LODWORD(a5) = 1;
        return (unsigned int)a5;
      }
      v24 = *(__int64 **)(a1 + 32);
      v25 = &v24[v16];
      while ( 2 )
      {
        v30 = *(_DWORD *)(a1 + 24);
        if ( !v30 )
        {
          ++*(_QWORD *)a1;
          v38[0] = 0;
          goto LABEL_27;
        }
        v26 = *(_QWORD *)(a1 + 8);
        v27 = (v30 - 1) & (((unsigned int)*v24 >> 9) ^ ((unsigned int)*v24 >> 4));
        v28 = (__int64 *)(v26 + 8LL * v27);
        v29 = *v28;
        if ( *v24 == *v28 )
        {
LABEL_24:
          if ( v25 == ++v24 )
            goto LABEL_18;
          continue;
        }
        break;
      }
      v36 = 1;
      v31 = 0;
      while ( v29 != -4096 )
      {
        if ( v29 != -8192 || v31 )
          v28 = v31;
        v27 = (v30 - 1) & (v36 + v27);
        v29 = *(_QWORD *)(v26 + 8LL * v27);
        if ( *v24 == v29 )
          goto LABEL_24;
        ++v36;
        v31 = v28;
        v28 = (__int64 *)(v26 + 8LL * v27);
      }
      v37 = *(_DWORD *)(a1 + 16);
      if ( !v31 )
        v31 = v28;
      ++*(_QWORD *)a1;
      v32 = v37 + 1;
      v38[0] = v31;
      if ( 4 * (v37 + 1) < 3 * v30 )
      {
        if ( v30 - *(_DWORD *)(a1 + 20) - v32 <= v30 >> 3 )
        {
LABEL_28:
          sub_CE2A30(a1, v30);
          sub_DA5B20(a1, v24, v38);
          v31 = (__int64 *)v38[0];
          v32 = *(_DWORD *)(a1 + 16) + 1;
        }
        *(_DWORD *)(a1 + 16) = v32;
        if ( *v31 != -4096 )
          --*(_DWORD *)(a1 + 20);
        *v31 = *v24;
        goto LABEL_24;
      }
LABEL_27:
      v30 *= 2;
      goto LABEL_28;
    }
    if ( *v10 == v11 )
      goto LABEL_9;
    ++v10;
  }
  if ( *v10 != v11 )
  {
    ++v10;
    goto LABEL_36;
  }
LABEL_9:
  a5 = 0;
  if ( v12 == v10 )
    goto LABEL_15;
  return (unsigned int)a5;
}
