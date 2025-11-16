// Function: sub_1BFE500
// Address: 0x1bfe500
//
__int64 *__fastcall sub_1BFE500(__int64 a1, __int64 a2)
{
  __int64 v3; // r12
  unsigned int v4; // r8d
  __int64 v5; // rdi
  __int64 v6; // rcx
  unsigned int v7; // r9d
  unsigned int v8; // esi
  __int64 *result; // rax
  __int64 v10; // r10
  __int64 *v11; // r11
  int v12; // r13d
  int v13; // esi
  int v14; // esi
  int v15; // ecx
  unsigned int v16; // esi
  __int64 v17; // r8
  unsigned int v18; // edx
  int v19; // edi
  __int64 v20; // rdx
  unsigned int v21; // r14d
  int v22; // r11d
  int v23; // r11d
  __int64 *v24; // r10
  int v25; // edx
  int v26; // edi
  int v27; // r11d
  __int64 *v28; // rdx
  int v29; // edx
  int v30; // r13d
  __int64 v31; // [rsp+8h] [rbp-38h] BYREF
  int v32; // [rsp+14h] [rbp-2Ch] BYREF
  _QWORD v33[5]; // [rsp+18h] [rbp-28h] BYREF

  v3 = a1 + 8;
  v4 = *(_DWORD *)(a1 + 32);
  v5 = *(_QWORD *)(a1 + 16);
  v31 = a2;
  if ( !v4 )
  {
    ++*(_QWORD *)(a1 + 8);
    v12 = *(_DWORD *)a1;
    goto LABEL_6;
  }
  v6 = a2;
  v7 = v4 - 1;
  v8 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  result = (__int64 *)(v5 + 16LL * v8);
  v10 = *result;
  v11 = result;
  if ( v6 != *result )
  {
    v20 = *result;
    v21 = v8;
    v22 = 1;
    while ( v20 != -8 )
    {
      v30 = v22 + 1;
      v21 = v7 & (v21 + v22);
      v11 = (__int64 *)(v5 + 16LL * v21);
      v20 = *v11;
      if ( v6 == *v11 )
        goto LABEL_3;
      v22 = v30;
    }
    v12 = *(_DWORD *)a1;
    v8 = v7 & (((unsigned int)v6 >> 4) ^ ((unsigned int)v6 >> 9));
    result = (__int64 *)(v5 + 16LL * v8);
    v10 = *result;
LABEL_12:
    if ( v6 == v10 )
      goto LABEL_13;
    v27 = 1;
    v28 = 0;
    while ( v10 != -8 )
    {
      if ( v10 == -16 && !v28 )
        v28 = result;
      v8 = v7 & (v27 + v8);
      result = (__int64 *)(v5 + 16LL * v8);
      v10 = *result;
      if ( v6 == *result )
        goto LABEL_13;
      ++v27;
    }
    if ( v28 )
      result = v28;
    v29 = *(_DWORD *)(a1 + 24);
    ++*(_QWORD *)(a1 + 8);
    v14 = v29 + 1;
    if ( 4 * (v29 + 1) < 3 * v4 )
    {
      if ( v4 - *(_DWORD *)(a1 + 28) - v14 > v4 >> 3 )
        goto LABEL_8;
      v13 = v4;
LABEL_7:
      sub_1BFE340(v3, v13);
      sub_1BFD9C0(v3, &v31, v33);
      result = (__int64 *)v33[0];
      v6 = v31;
      v14 = *(_DWORD *)(a1 + 24) + 1;
LABEL_8:
      *(_DWORD *)(a1 + 24) = v14;
      if ( *result != -8 )
        --*(_DWORD *)(a1 + 28);
      *result = v6;
      *((_DWORD *)result + 2) = 0;
LABEL_13:
      *((_DWORD *)result + 2) = v12;
      v15 = *(_DWORD *)a1;
      v16 = *(_DWORD *)(a1 + 64);
      v32 = *(_DWORD *)a1;
      if ( v16 )
      {
        v17 = *(_QWORD *)(a1 + 48);
        v18 = (v16 - 1) & (37 * v15);
        result = (__int64 *)(v17 + 16LL * v18);
        v19 = *(_DWORD *)result;
        if ( v15 == *(_DWORD *)result )
        {
LABEL_15:
          result[1] = v31;
          ++*(_DWORD *)a1;
          return result;
        }
        v23 = 1;
        v24 = 0;
        while ( v19 != 0x7FFFFFFF )
        {
          if ( v19 == 0x80000000 && !v24 )
            v24 = result;
          v18 = (v16 - 1) & (v23 + v18);
          result = (__int64 *)(v17 + 16LL * v18);
          v19 = *(_DWORD *)result;
          if ( v15 == *(_DWORD *)result )
            goto LABEL_15;
          ++v23;
        }
        v25 = *(_DWORD *)(a1 + 56);
        if ( v24 )
          result = v24;
        ++*(_QWORD *)(a1 + 40);
        v26 = v25 + 1;
        if ( 4 * (v25 + 1) < 3 * v16 )
        {
          if ( v16 - *(_DWORD *)(a1 + 60) - v26 > v16 >> 3 )
          {
LABEL_25:
            *(_DWORD *)(a1 + 56) = v26;
            if ( *(_DWORD *)result != 0x7FFFFFFF )
              --*(_DWORD *)(a1 + 60);
            *(_DWORD *)result = v15;
            result[1] = 0;
            goto LABEL_15;
          }
LABEL_30:
          sub_1A64A90(a1 + 40, v16);
          sub_1BFD7C0(a1 + 40, &v32, v33);
          result = (__int64 *)v33[0];
          v15 = v32;
          v26 = *(_DWORD *)(a1 + 56) + 1;
          goto LABEL_25;
        }
      }
      else
      {
        ++*(_QWORD *)(a1 + 40);
      }
      v16 *= 2;
      goto LABEL_30;
    }
LABEL_6:
    v13 = 2 * v4;
    goto LABEL_7;
  }
LABEL_3:
  if ( v11 == (__int64 *)(v5 + 16LL * v4) )
  {
    v12 = *(_DWORD *)a1;
    goto LABEL_12;
  }
  return result;
}
