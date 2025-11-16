// Function: sub_2A6FB70
// Address: 0x2a6fb70
//
unsigned __int64 __fastcall sub_2A6FB70(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r12
  unsigned int v5; // esi
  __int64 v6; // rcx
  __int64 v7; // r9
  __int64 v8; // r8
  int v9; // r10d
  unsigned int v10; // edx
  __int64 v11; // r13
  __int64 v12; // rax
  __int64 v13; // rdi
  unsigned __int64 result; // rax
  __int64 v15; // r12
  unsigned int v16; // esi
  __int64 v17; // rcx
  __int64 v18; // r9
  __int64 *v19; // r11
  int v20; // ebx
  unsigned int v21; // edx
  __int64 *v22; // rdi
  __int64 v23; // r8
  int v24; // edi
  int v25; // edi
  __int64 v26; // rcx
  _QWORD *v27; // rsi
  __int64 v28; // rdi
  _QWORD *v29; // rdx
  __int64 *v30; // rbx
  __int64 *v31; // r14
  __int64 v32; // r8
  _QWORD *v33; // rdi
  __int64 v34; // rcx
  unsigned int v35; // esi
  int v36; // edx
  __int64 *v37; // r14
  int v38; // eax
  __int64 *v39; // rax
  __int64 v40; // rbx
  int v41; // r11d
  _QWORD *v42; // r10
  int v43; // eax
  __int64 v44; // [rsp+0h] [rbp-50h] BYREF
  __int64 v45; // [rsp+8h] [rbp-48h] BYREF
  _QWORD v46[7]; // [rsp+18h] [rbp-38h] BYREF

  v3 = a1 + 2568;
  v45 = a2;
  v5 = *(_DWORD *)(a1 + 2592);
  v44 = a3;
  if ( !v5 )
  {
    ++*(_QWORD *)(a1 + 2568);
    v46[0] = 0;
    goto LABEL_54;
  }
  v6 = v45;
  v7 = v5 - 1;
  v8 = *(_QWORD *)(a1 + 2576);
  v9 = 1;
  v10 = v7 & (((unsigned int)v45 >> 9) ^ ((unsigned int)v45 >> 4));
  v11 = v8 + 72LL * v10;
  v12 = 0;
  v13 = *(_QWORD *)v11;
  if ( v45 != *(_QWORD *)v11 )
  {
    while ( v13 != -4096 )
    {
      if ( !v12 && v13 == -8192 )
        v12 = v11;
      v10 = v7 & (v9 + v10);
      v11 = v8 + 72LL * v10;
      v13 = *(_QWORD *)v11;
      if ( v45 == *(_QWORD *)v11 )
        goto LABEL_3;
      ++v9;
    }
    v24 = *(_DWORD *)(a1 + 2584);
    if ( !v12 )
      v12 = v11;
    ++*(_QWORD *)(a1 + 2568);
    v25 = v24 + 1;
    v46[0] = v12;
    if ( 4 * v25 < 3 * v5 )
    {
      v8 = v5 >> 3;
      if ( v5 - *(_DWORD *)(a1 + 2588) - v25 > (unsigned int)v8 )
      {
LABEL_17:
        *(_DWORD *)(a1 + 2584) = v25;
        if ( *(_QWORD *)v12 != -4096 )
          --*(_DWORD *)(a1 + 2588);
        *(_QWORD *)v12 = v6;
        v15 = v12 + 8;
        *(_QWORD *)(v12 + 40) = v12 + 56;
        *(_QWORD *)(v12 + 48) = 0x200000000LL;
        *(_OWORD *)(v12 + 8) = 0;
        *(_OWORD *)(v12 + 24) = 0;
        *(_OWORD *)(v12 + 56) = 0;
        goto LABEL_20;
      }
LABEL_55:
      sub_2A69530(v3, v5);
      sub_2A657F0(v3, &v45, v46);
      v6 = v45;
      v25 = *(_DWORD *)(a1 + 2584) + 1;
      v12 = v46[0];
      goto LABEL_17;
    }
LABEL_54:
    v5 *= 2;
    goto LABEL_55;
  }
LABEL_3:
  result = *(unsigned int *)(v11 + 24);
  v15 = v11 + 8;
  if ( (_DWORD)result )
  {
    v16 = *(_DWORD *)(v11 + 32);
    if ( v16 )
    {
      v17 = v44;
      v18 = *(_QWORD *)(v11 + 16);
      v19 = 0;
      v20 = 1;
      v21 = (v16 - 1) & (((unsigned int)v44 >> 9) ^ ((unsigned int)v44 >> 4));
      v22 = (__int64 *)(v18 + 8LL * v21);
      v23 = *v22;
      if ( v44 == *v22 )
        return result;
      while ( v23 != -4096 )
      {
        if ( v23 != -8192 || v19 )
          v22 = v19;
        v21 = (v16 - 1) & (v20 + v21);
        v37 = (__int64 *)(v18 + 8LL * v21);
        v23 = *v37;
        if ( v44 == *v37 )
          return result;
        ++v20;
        v19 = v22;
        v22 = (__int64 *)(v18 + 8LL * v21);
      }
      if ( !v19 )
        v19 = v22;
      v38 = result + 1;
      v46[0] = v19;
      ++*(_QWORD *)(v11 + 8);
      if ( 4 * v38 < 3 * v16 )
      {
        if ( v16 - (v38 + *(_DWORD *)(v11 + 28)) > v16 >> 3 )
        {
LABEL_69:
          *(_DWORD *)(v11 + 24) = v38;
          v39 = (__int64 *)v46[0];
          if ( *(_QWORD *)v46[0] != -4096 )
            --*(_DWORD *)(v11 + 28);
          *v39 = v17;
          result = *(unsigned int *)(v11 + 48);
          v40 = v44;
          if ( result + 1 > *(unsigned int *)(v11 + 52) )
          {
            sub_C8D5F0(v11 + 40, (const void *)(v11 + 56), result + 1, 8u, v23, v18);
            result = *(unsigned int *)(v11 + 48);
          }
          *(_QWORD *)(*(_QWORD *)(v11 + 40) + 8 * result) = v40;
          ++*(_DWORD *)(v11 + 48);
          return result;
        }
LABEL_83:
        sub_27D4930(v11 + 8, v16);
        sub_2A68360(v11 + 8, &v44, v46);
        v17 = v44;
        v38 = *(_DWORD *)(v11 + 24) + 1;
        goto LABEL_69;
      }
    }
    else
    {
      v46[0] = 0;
      ++*(_QWORD *)(v11 + 8);
    }
    v16 *= 2;
    goto LABEL_83;
  }
LABEL_20:
  v26 = *(unsigned int *)(v15 + 40);
  result = *(_QWORD *)(v15 + 32);
  v27 = (_QWORD *)(result + 8 * v26);
  v28 = (8 * v26) >> 3;
  if ( !((8 * v26) >> 5) )
    goto LABEL_42;
  v29 = (_QWORD *)(result + 32 * ((8 * v26) >> 5));
  while ( 1 )
  {
    if ( *(_QWORD *)result == v44 )
      goto LABEL_27;
    if ( *(_QWORD *)(result + 8) == v44 )
    {
      result += 8LL;
      if ( v27 == (_QWORD *)result )
        goto LABEL_28;
      return result;
    }
    if ( *(_QWORD *)(result + 16) == v44 )
    {
      result += 16LL;
      if ( v27 == (_QWORD *)result )
        goto LABEL_28;
      return result;
    }
    if ( *(_QWORD *)(result + 24) == v44 )
      break;
    result += 32LL;
    if ( v29 == (_QWORD *)result )
    {
      v28 = (__int64)((__int64)v27 - result) >> 3;
LABEL_42:
      if ( v28 == 2 )
        goto LABEL_58;
      if ( v28 != 3 )
      {
        if ( v28 == 1 )
          goto LABEL_45;
        goto LABEL_28;
      }
      if ( *(_QWORD *)result == v44 )
        goto LABEL_27;
      result += 8LL;
LABEL_58:
      if ( *(_QWORD *)result == v44 )
        goto LABEL_27;
      result += 8LL;
LABEL_45:
      if ( *(_QWORD *)result == v44 )
      {
LABEL_27:
        if ( v27 == (_QWORD *)result )
          goto LABEL_28;
        return result;
      }
LABEL_28:
      if ( v26 + 1 > (unsigned __int64)*(unsigned int *)(v15 + 44) )
      {
        sub_C8D5F0(v15 + 32, (const void *)(v15 + 48), v26 + 1, 8u, v8, v7);
        v27 = (_QWORD *)(*(_QWORD *)(v15 + 32) + 8LL * *(unsigned int *)(v15 + 40));
      }
      *v27 = v44;
      result = (unsigned int)(*(_DWORD *)(v15 + 40) + 1);
      *(_DWORD *)(v15 + 40) = result;
      if ( (unsigned int)result <= 2 )
        return result;
      v30 = *(__int64 **)(v15 + 32);
      v31 = &v30[result];
      while ( 2 )
      {
        v35 = *(_DWORD *)(v15 + 24);
        if ( !v35 )
        {
          v46[0] = 0;
          ++*(_QWORD *)v15;
          goto LABEL_36;
        }
        v32 = *(_QWORD *)(v15 + 8);
        result = (v35 - 1) & (((unsigned int)*v30 >> 9) ^ ((unsigned int)*v30 >> 4));
        v33 = (_QWORD *)(v32 + 8 * result);
        v34 = *v33;
        if ( *v33 == *v30 )
        {
LABEL_33:
          if ( v31 == ++v30 )
            return result;
          continue;
        }
        break;
      }
      v41 = 1;
      v42 = 0;
      while ( v34 != -4096 )
      {
        if ( v42 || v34 != -8192 )
          v33 = v42;
        result = (v35 - 1) & (v41 + (_DWORD)result);
        v34 = *(_QWORD *)(v32 + 8LL * (unsigned int)result);
        if ( *v30 == v34 )
          goto LABEL_33;
        ++v41;
        v42 = v33;
        v33 = (_QWORD *)(v32 + 8LL * (unsigned int)result);
      }
      if ( !v42 )
        v42 = v33;
      v46[0] = v42;
      v43 = *(_DWORD *)(v15 + 16);
      ++*(_QWORD *)v15;
      v36 = v43 + 1;
      if ( 4 * (v43 + 1) < 3 * v35 )
      {
        if ( v35 - *(_DWORD *)(v15 + 20) - v36 <= v35 >> 3 )
        {
LABEL_37:
          sub_27D4930(v15, v35);
          sub_2A68360(v15, v30, v46);
          v36 = *(_DWORD *)(v15 + 16) + 1;
        }
        *(_DWORD *)(v15 + 16) = v36;
        result = v46[0];
        if ( *(_QWORD *)v46[0] != -4096 )
          --*(_DWORD *)(v15 + 20);
        *(_QWORD *)result = *v30;
        goto LABEL_33;
      }
LABEL_36:
      v35 *= 2;
      goto LABEL_37;
    }
  }
  result += 24LL;
  if ( v27 == (_QWORD *)result )
    goto LABEL_28;
  return result;
}
