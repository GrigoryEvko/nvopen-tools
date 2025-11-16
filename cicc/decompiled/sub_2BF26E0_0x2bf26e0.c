// Function: sub_2BF26E0
// Address: 0x2bf26e0
//
__int64 *__fastcall sub_2BF26E0(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  unsigned int v6; // esi
  __int64 v7; // r13
  __int64 v8; // r8
  __int64 v9; // rdi
  __int64 *v10; // r10
  int v11; // r14d
  unsigned int v12; // ecx
  __int64 *v13; // rax
  __int64 v14; // rdx
  __int64 *result; // rax
  __int64 v16; // r8
  __int64 v17; // rdx
  __int64 v18; // r13
  __int64 v19; // r9
  __int64 v20; // rdi
  int v21; // r11d
  unsigned int v22; // ecx
  __int64 v23; // rsi
  __int64 v24; // r10
  __int64 v25; // rbx
  __int64 *v26; // rdx
  int v27; // esi
  int v28; // esi
  __int64 v29; // rax
  __int64 v30; // rcx
  int v31; // edx
  int v32; // eax
  int v33; // esi
  __int64 v34; // [rsp+8h] [rbp-38h] BYREF
  __int64 v35; // [rsp+10h] [rbp-30h] BYREF
  _QWORD v36[5]; // [rsp+18h] [rbp-28h] BYREF

  v34 = a2;
  if ( !a4 )
  {
    v6 = *(_DWORD *)(a1 + 56);
    v7 = a1 + 32;
    if ( v6 )
    {
      v8 = *(_QWORD *)(a1 + 40);
      v9 = v34;
      v10 = 0;
      v11 = 1;
      v12 = (v6 - 1) & (((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4));
      v13 = (__int64 *)(v8 + 16LL * v12);
      v14 = *v13;
      if ( v34 == *v13 )
      {
LABEL_4:
        result = v13 + 1;
LABEL_5:
        *result = a3;
        return result;
      }
      while ( v14 != -4096 )
      {
        if ( !v10 && v14 == -8192 )
          v10 = v13;
        v12 = (v6 - 1) & (v11 + v12);
        v13 = (__int64 *)(v8 + 16LL * v12);
        v14 = *v13;
        if ( v34 == *v13 )
          goto LABEL_4;
        ++v11;
      }
      if ( !v10 )
        v10 = v13;
      v32 = *(_DWORD *)(a1 + 48);
      ++*(_QWORD *)(a1 + 32);
      v31 = v32 + 1;
      v36[0] = v10;
      if ( 4 * (v32 + 1) < 3 * v6 )
      {
        if ( v6 - *(_DWORD *)(a1 + 52) - v31 > v6 >> 3 )
          goto LABEL_35;
        goto LABEL_34;
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 32);
      v36[0] = 0;
    }
    v6 *= 2;
LABEL_34:
    sub_2AC6AB0(v7, v6);
    sub_2ABE290(v7, &v34, v36);
    v9 = v34;
    v10 = (__int64 *)v36[0];
    v31 = *(_DWORD *)(a1 + 48) + 1;
LABEL_35:
    *(_DWORD *)(a1 + 48) = v31;
    if ( *v10 != -4096 )
      --*(_DWORD *)(a1 + 52);
    *v10 = v9;
    result = v10 + 1;
    v10[1] = 0;
    goto LABEL_5;
  }
  v16 = *(unsigned int *)(a1 + 88);
  v35 = a2;
  v17 = a2;
  v18 = a1 + 64;
  if ( !(_DWORD)v16 )
  {
    ++*(_QWORD *)(a1 + 64);
    v36[0] = 0;
    goto LABEL_51;
  }
  v19 = (unsigned int)(v16 - 1);
  v20 = *(_QWORD *)(a1 + 72);
  v21 = 1;
  v22 = v19 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v23 = v20 + 56LL * v22;
  result = 0;
  v24 = *(_QWORD *)v23;
  if ( v17 != *(_QWORD *)v23 )
  {
    while ( v24 != -4096 )
    {
      if ( !result && v24 == -8192 )
        result = (__int64 *)v23;
      v22 = v19 & (v21 + v22);
      v23 = v20 + 56LL * v22;
      v24 = *(_QWORD *)v23;
      if ( v17 == *(_QWORD *)v23 )
        goto LABEL_8;
      ++v21;
    }
    if ( !result )
      result = (__int64 *)v23;
    v27 = *(_DWORD *)(a1 + 80);
    ++*(_QWORD *)(a1 + 64);
    v28 = v27 + 1;
    v36[0] = result;
    if ( 4 * v28 < (unsigned int)(3 * v16) )
    {
      if ( (int)v16 - *(_DWORD *)(a1 + 84) - v28 > (unsigned int)v16 >> 3 )
      {
LABEL_21:
        *(_DWORD *)(a1 + 80) = v28;
        if ( *result != -4096 )
          --*(_DWORD *)(a1 + 84);
        *result = v17;
        result[1] = (__int64)(result + 3);
        result[2] = 0x400000000LL;
        v25 = (__int64)(result + 1);
        goto LABEL_24;
      }
      v33 = v16;
LABEL_52:
      sub_2AC6C60(v18, v33);
      sub_2ABE350(v18, &v35, v36);
      v17 = v35;
      v28 = *(_DWORD *)(a1 + 80) + 1;
      result = (__int64 *)v36[0];
      goto LABEL_21;
    }
LABEL_51:
    v33 = 2 * v16;
    goto LABEL_52;
  }
LABEL_8:
  v25 = v23 + 8;
  if ( !*(_DWORD *)(v23 + 16) )
  {
LABEL_24:
    if ( *(_DWORD *)(v25 + 12) )
    {
      v29 = 0;
    }
    else
    {
      sub_C8D5F0(v25, (const void *)(v25 + 16), 1u, 8u, v16, v19);
      v29 = *(unsigned int *)(v25 + 8);
    }
    v26 = *(__int64 **)v25;
    result = (__int64 *)(*(_QWORD *)v25 + 8 * v29);
    v30 = *(_QWORD *)v25 + 8LL;
    if ( result != (__int64 *)v30 )
    {
      do
      {
        if ( result )
          *result = 0;
        ++result;
      }
      while ( (__int64 *)v30 != result );
      v26 = *(__int64 **)v25;
    }
    *(_DWORD *)(v25 + 8) = 1;
    goto LABEL_10;
  }
  v26 = *(__int64 **)(v23 + 8);
LABEL_10:
  *v26 = a3;
  return result;
}
