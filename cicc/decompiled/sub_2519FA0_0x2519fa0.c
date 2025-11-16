// Function: sub_2519FA0
// Address: 0x2519fa0
//
__int64 __fastcall sub_2519FA0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  __int64 result; // rax
  __int64 v9; // rcx
  __int64 *v10; // rsi
  __int64 v11; // rdi
  __int64 v12; // rdx
  __int64 *v13; // r12
  __int64 *v14; // r14
  __int64 v15; // r8
  __int64 *v16; // rdi
  __int64 v17; // rcx
  unsigned int v18; // esi
  __int64 *v19; // r10
  int v20; // edx
  unsigned int v21; // esi
  __int64 v22; // r13
  __int64 v23; // r9
  int v24; // r11d
  __int64 v25; // r8
  __int64 *v26; // r10
  unsigned int v27; // edx
  __int64 *v28; // rcx
  __int64 v29; // rdi
  int v30; // eax
  __int64 v31; // r12
  int v32; // r11d
  int v33; // eax
  __int64 v34; // [rsp+10h] [rbp-40h] BYREF
  _QWORD v35[7]; // [rsp+18h] [rbp-38h] BYREF

  v6 = a2;
  result = *(unsigned int *)(a1 + 4136);
  v34 = a2;
  if ( (_DWORD)result )
  {
    v21 = *(_DWORD *)(a1 + 4144);
    v22 = a1 + 4120;
    if ( v21 )
    {
      v23 = v21 - 1;
      v24 = 1;
      v25 = *(_QWORD *)(a1 + 4128);
      v26 = 0;
      v27 = v23 & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
      v28 = (__int64 *)(v25 + 8LL * v27);
      v29 = *v28;
      if ( v6 == *v28 )
        return result;
      while ( v29 != -4096 )
      {
        if ( v29 != -8192 || v26 )
          v28 = v26;
        v27 = v23 & (v24 + v27);
        v29 = *(_QWORD *)(v25 + 8LL * v27);
        if ( v6 == v29 )
          return result;
        ++v24;
        v26 = v28;
        v28 = (__int64 *)(v25 + 8LL * v27);
      }
      if ( !v26 )
        v26 = v28;
      v30 = result + 1;
      ++*(_QWORD *)(a1 + 4120);
      v35[0] = v26;
      if ( 4 * v30 < 3 * v21 )
      {
        if ( v21 - *(_DWORD *)(a1 + 4140) - v30 > v21 >> 3 )
        {
LABEL_35:
          *(_DWORD *)(a1 + 4136) = v30;
          if ( *v26 != -4096 )
            --*(_DWORD *)(a1 + 4140);
          *v26 = v6;
          result = *(unsigned int *)(a1 + 4160);
          v31 = v34;
          if ( result + 1 > (unsigned __int64)*(unsigned int *)(a1 + 4164) )
          {
            sub_C8D5F0(a1 + 4152, (const void *)(a1 + 4168), result + 1, 8u, v25, v23);
            result = *(unsigned int *)(a1 + 4160);
          }
          *(_QWORD *)(*(_QWORD *)(a1 + 4152) + 8 * result) = v31;
          ++*(_DWORD *)(a1 + 4160);
          return result;
        }
LABEL_61:
        sub_2519C60(v22, v21);
        sub_2512640(v22, &v34, v35);
        v6 = v34;
        v26 = (__int64 *)v35[0];
        v30 = *(_DWORD *)(a1 + 4136) + 1;
        goto LABEL_35;
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 4120);
      v35[0] = 0;
    }
    v21 *= 2;
    goto LABEL_61;
  }
  v9 = *(unsigned int *)(a1 + 4160);
  result = *(_QWORD *)(a1 + 4152);
  v10 = (__int64 *)(result + 8 * v9);
  v11 = (8 * v9) >> 3;
  if ( !((8 * v9) >> 5) )
    goto LABEL_12;
  v12 = result + 32 * ((8 * v9) >> 5);
  do
  {
    if ( v6 == *(_QWORD *)result )
      goto LABEL_9;
    if ( v6 == *(_QWORD *)(result + 8) )
    {
      result += 8;
      if ( v10 == (__int64 *)result )
        goto LABEL_15;
      return result;
    }
    if ( v6 == *(_QWORD *)(result + 16) )
    {
      result += 16;
      if ( v10 == (__int64 *)result )
        goto LABEL_15;
      return result;
    }
    if ( v6 == *(_QWORD *)(result + 24) )
    {
      result += 24;
      if ( v10 == (__int64 *)result )
        goto LABEL_15;
      return result;
    }
    result += 32;
  }
  while ( v12 != result );
  v11 = ((__int64)v10 - result) >> 3;
LABEL_12:
  if ( v11 == 2 )
  {
LABEL_42:
    if ( v6 != *(_QWORD *)result )
    {
      result += 8;
      goto LABEL_44;
    }
LABEL_9:
    if ( v10 == (__int64 *)result )
      goto LABEL_15;
    return result;
  }
  if ( v11 == 3 )
  {
    if ( v6 == *(_QWORD *)result )
      goto LABEL_9;
    result += 8;
    goto LABEL_42;
  }
  if ( v11 != 1 )
    goto LABEL_15;
LABEL_44:
  if ( v6 == *(_QWORD *)result )
    goto LABEL_9;
LABEL_15:
  if ( v9 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 4164) )
  {
    sub_C8D5F0(a1 + 4152, (const void *)(a1 + 4168), v9 + 1, 8u, a5, a6);
    v10 = (__int64 *)(*(_QWORD *)(a1 + 4152) + 8LL * *(unsigned int *)(a1 + 4160));
  }
  *v10 = v6;
  result = (unsigned int)(*(_DWORD *)(a1 + 4160) + 1);
  *(_DWORD *)(a1 + 4160) = result;
  if ( (unsigned int)result > 0x10 )
  {
    v13 = *(__int64 **)(a1 + 4152);
    v14 = &v13[result];
    while ( 1 )
    {
      v18 = *(_DWORD *)(a1 + 4144);
      if ( !v18 )
        break;
      v15 = *(_QWORD *)(a1 + 4128);
      result = (v18 - 1) & (((unsigned int)*v13 >> 9) ^ ((unsigned int)*v13 >> 4));
      v16 = (__int64 *)(v15 + 8 * result);
      v17 = *v16;
      if ( *v13 != *v16 )
      {
        v32 = 1;
        v19 = 0;
        while ( v17 != -4096 )
        {
          if ( v19 || v17 != -8192 )
            v16 = v19;
          result = (v18 - 1) & (v32 + (_DWORD)result);
          v17 = *(_QWORD *)(v15 + 8LL * (unsigned int)result);
          if ( *v13 == v17 )
            goto LABEL_20;
          ++v32;
          v19 = v16;
          v16 = (__int64 *)(v15 + 8LL * (unsigned int)result);
        }
        v33 = *(_DWORD *)(a1 + 4136);
        if ( !v19 )
          v19 = v16;
        ++*(_QWORD *)(a1 + 4120);
        v20 = v33 + 1;
        v35[0] = v19;
        if ( 4 * (v33 + 1) < 3 * v18 )
        {
          if ( v18 - *(_DWORD *)(a1 + 4140) - v20 <= v18 >> 3 )
          {
LABEL_24:
            sub_2519C60(a1 + 4120, v18);
            sub_2512640(a1 + 4120, v13, v35);
            v19 = (__int64 *)v35[0];
            v20 = *(_DWORD *)(a1 + 4136) + 1;
          }
          *(_DWORD *)(a1 + 4136) = v20;
          if ( *v19 != -4096 )
            --*(_DWORD *)(a1 + 4140);
          result = *v13;
          *v19 = *v13;
          goto LABEL_20;
        }
LABEL_23:
        v18 *= 2;
        goto LABEL_24;
      }
LABEL_20:
      if ( v14 == ++v13 )
        return result;
    }
    ++*(_QWORD *)(a1 + 4120);
    v35[0] = 0;
    goto LABEL_23;
  }
  return result;
}
