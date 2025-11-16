// Function: sub_FFACA0
// Address: 0xffaca0
//
unsigned __int64 __fastcall sub_FFACA0(__int64 a1, __int64 a2)
{
  __int64 v4; // rdi
  __int64 v5; // rsi
  _BYTE *v6; // r12
  unsigned __int64 result; // rax
  _BYTE *v8; // r10
  __int64 v9; // r14
  __int64 v10; // r9
  __int64 v11; // r8
  unsigned int v12; // edi
  __int64 v13; // rax
  __int64 v14; // rcx
  __int64 v15; // rdx
  _QWORD *v16; // rcx
  __int64 v17; // r14
  __int64 v18; // rdi
  __int64 *v19; // rdx
  int v20; // eax
  int v21; // eax
  unsigned __int64 v22; // rdx
  int v23; // esi
  unsigned int v24; // ecx
  __int64 v25; // rdi
  int v26; // r15d
  int v27; // ecx
  int v28; // ecx
  __int64 v29; // rdi
  unsigned int v30; // r15d
  int v31; // [rsp+10h] [rbp-E0h]
  _BYTE *v32; // [rsp+10h] [rbp-E0h]
  _BYTE *v33; // [rsp+10h] [rbp-E0h]
  _BYTE *v34; // [rsp+10h] [rbp-E0h]
  _BYTE **v35; // [rsp+28h] [rbp-C8h] BYREF
  _BYTE *v36; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v37; // [rsp+38h] [rbp-B8h]
  _BYTE v38[176]; // [rsp+40h] [rbp-B0h] BYREF

  v4 = *(_QWORD *)(a2 - 96);
  v36 = v38;
  v5 = 0;
  v37 = 0x1000000000LL;
  v35 = &v36;
  sub_997910(v4, 0, (__int64 (__fastcall *)(__int64, unsigned __int8 *))sub_FFA910, (__int64)&v35);
  v6 = v36;
  result = (unsigned int)v37;
  v8 = &v36[8 * (unsigned int)v37];
  if ( v8 == v36 )
    goto LABEL_14;
  do
  {
    v5 = *(unsigned int *)(a1 + 24);
    v9 = *(_QWORD *)v6;
    if ( !(_DWORD)v5 )
    {
      ++*(_QWORD *)a1;
      goto LABEL_44;
    }
    v10 = (unsigned int)(v5 - 1);
    v11 = *(_QWORD *)(a1 + 8);
    v12 = v10 & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
    v13 = v11 + 32LL * v12;
    v14 = *(_QWORD *)v13;
    if ( v9 != *(_QWORD *)v13 )
    {
      v31 = 1;
      v19 = 0;
      while ( v14 != -4096 )
      {
        if ( !v19 && v14 == -8192 )
          v19 = (__int64 *)v13;
        v12 = v10 & (v31 + v12);
        v11 = (unsigned int)(v31 + 1);
        v13 = *(_QWORD *)(a1 + 8) + 32LL * v12;
        v14 = *(_QWORD *)v13;
        if ( v9 == *(_QWORD *)v13 )
          goto LABEL_4;
        ++v31;
      }
      if ( !v19 )
        v19 = (__int64 *)v13;
      v20 = *(_DWORD *)(a1 + 16);
      ++*(_QWORD *)a1;
      v21 = v20 + 1;
      if ( 4 * v21 < (unsigned int)(3 * v5) )
      {
        if ( (int)v5 - *(_DWORD *)(a1 + 20) - v21 <= (unsigned int)v5 >> 3 )
        {
          v34 = v8;
          sub_FFA970(a1, v5);
          v27 = *(_DWORD *)(a1 + 24);
          if ( !v27 )
          {
LABEL_72:
            ++*(_DWORD *)(a1 + 16);
            BUG();
          }
          v28 = v27 - 1;
          v29 = *(_QWORD *)(a1 + 8);
          v11 = 0;
          v30 = v28 & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
          v10 = 1;
          v8 = v34;
          v21 = *(_DWORD *)(a1 + 16) + 1;
          v19 = (__int64 *)(v29 + 32LL * v30);
          v5 = *v19;
          if ( v9 != *v19 )
          {
            while ( v5 != -4096 )
            {
              if ( v5 == -8192 && !v11 )
                v11 = (__int64)v19;
              v30 = v28 & (v10 + v30);
              v19 = (__int64 *)(v29 + 32LL * v30);
              v5 = *v19;
              if ( v9 == *v19 )
                goto LABEL_23;
              v10 = (unsigned int)(v10 + 1);
            }
            if ( v11 )
              v19 = (__int64 *)v11;
          }
        }
        goto LABEL_23;
      }
LABEL_44:
      v33 = v8;
      sub_FFA970(a1, 2 * v5);
      v23 = *(_DWORD *)(a1 + 24);
      if ( !v23 )
        goto LABEL_72;
      v5 = (unsigned int)(v23 - 1);
      v11 = *(_QWORD *)(a1 + 8);
      v8 = v33;
      v24 = v5 & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
      v21 = *(_DWORD *)(a1 + 16) + 1;
      v19 = (__int64 *)(v11 + 32LL * v24);
      v25 = *v19;
      if ( v9 != *v19 )
      {
        v26 = 1;
        v10 = 0;
        while ( v25 != -4096 )
        {
          if ( !v10 && v25 == -8192 )
            v10 = (__int64)v19;
          v24 = v5 & (v26 + v24);
          v19 = (__int64 *)(v11 + 32LL * v24);
          v25 = *v19;
          if ( v9 == *v19 )
            goto LABEL_23;
          ++v26;
        }
        if ( v10 )
          v19 = (__int64 *)v10;
      }
LABEL_23:
      *(_DWORD *)(a1 + 16) = v21;
      if ( *v19 != -4096 )
        --*(_DWORD *)(a1 + 20);
      v16 = v19 + 3;
      *v19 = v9;
      v17 = (__int64)(v19 + 1);
      v19[1] = (__int64)(v19 + 3);
      v19[2] = 0x100000000LL;
      v15 = 0;
      goto LABEL_26;
    }
LABEL_4:
    v15 = *(unsigned int *)(v13 + 16);
    v16 = *(_QWORD **)(v13 + 8);
    v17 = v13 + 8;
    v5 = (__int64)&v16[v15];
    v18 = (8 * v15) >> 3;
    result = (8 * v15) >> 5;
    if ( result )
    {
      while ( a2 != *v16 )
      {
        if ( a2 == v16[1] )
        {
          ++v16;
          break;
        }
        if ( a2 == v16[2] )
        {
          v16 += 2;
          break;
        }
        if ( a2 == v16[3] )
        {
          v16 += 3;
          break;
        }
        v16 += 4;
        if ( !--result )
        {
          v18 = (v5 - (__int64)v16) >> 3;
          goto LABEL_32;
        }
      }
LABEL_11:
      if ( v16 != (_QWORD *)v5 )
        goto LABEL_12;
LABEL_26:
      result = *(unsigned int *)(v17 + 12);
      v22 = v15 + 1;
      if ( v22 > result )
        goto LABEL_37;
      goto LABEL_27;
    }
LABEL_32:
    if ( v18 != 2 )
    {
      if ( v18 != 3 )
      {
        if ( v18 != 1 )
          goto LABEL_36;
        goto LABEL_35;
      }
      if ( a2 == *v16 )
        goto LABEL_11;
      ++v16;
    }
    if ( a2 == *v16 )
      goto LABEL_11;
    ++v16;
LABEL_35:
    if ( a2 == *v16 )
      goto LABEL_11;
LABEL_36:
    result = *(unsigned int *)(v17 + 12);
    v22 = v15 + 1;
    v16 = (_QWORD *)v5;
    if ( v22 > result )
    {
LABEL_37:
      v5 = v17 + 16;
      v32 = v8;
      sub_C8D5F0(v17, (const void *)(v17 + 16), v22, 8u, v11, v10);
      result = *(_QWORD *)v17;
      v8 = v32;
      v16 = (_QWORD *)(*(_QWORD *)v17 + 8LL * *(unsigned int *)(v17 + 8));
    }
LABEL_27:
    *v16 = a2;
    ++*(_DWORD *)(v17 + 8);
LABEL_12:
    v6 += 8;
  }
  while ( v8 != v6 );
  v6 = v36;
LABEL_14:
  if ( v6 != v38 )
    return _libc_free(v6, v5);
  return result;
}
