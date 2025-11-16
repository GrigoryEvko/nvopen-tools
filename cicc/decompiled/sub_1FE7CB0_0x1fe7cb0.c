// Function: sub_1FE7CB0
// Address: 0x1fe7cb0
//
__int64 __fastcall sub_1FE7CB0(__int64 a1, __int64 a2, unsigned __int64 *a3, _DWORD *a4)
{
  __int64 result; // rax
  __int64 v8; // rsi
  __int64 v9; // r15
  unsigned __int64 v10; // rcx
  int v11; // r9d
  __int64 v12; // r10
  __int64 v13; // rdi
  unsigned int v14; // edx
  __int64 *v15; // r8
  __int64 v16; // r14
  int v17; // r11d
  int v18; // edx
  int v19; // ecx
  int v20; // r8d
  __int64 v21; // r11
  __int64 v22; // r9
  int v23; // r14d
  unsigned int i; // edx
  unsigned int v25; // edx
  int v26; // ecx
  unsigned __int64 v27; // rdx
  __int64 v28; // rcx
  int v29; // edx
  int v30; // r9d
  int v31; // esi
  int v32; // r11d
  __int64 v33; // r8
  unsigned int j; // edx
  unsigned int v35; // edx
  int v36; // r10d
  int v37; // r10d
  int v38; // [rsp+8h] [rbp-38h]
  __int64 v39; // [rsp+8h] [rbp-38h]
  __int64 v40; // [rsp+8h] [rbp-38h]

  result = a1;
  v8 = *(unsigned int *)(a2 + 24);
  v9 = *(_QWORD *)a2;
  if ( !(_DWORD)v8 )
  {
    *(_QWORD *)a2 = v9 + 1;
    goto LABEL_14;
  }
  v10 = *a3;
  v11 = *((_DWORD *)a3 + 2);
  v38 = 1;
  v12 = *(_QWORD *)(a2 + 8);
  v13 = 0;
  v14 = (v8 - 1) & (v11 + ((v10 >> 9) ^ (v10 >> 4)));
  while ( 1 )
  {
    v15 = (__int64 *)(v12 + 24LL * v14);
    v16 = *v15;
    if ( v10 == *v15 && v11 == *((_DWORD *)v15 + 2) )
    {
      *(_QWORD *)result = a2;
      *(_QWORD *)(result + 8) = v9;
      *(_QWORD *)(result + 16) = v15;
      *(_QWORD *)(result + 24) = v12 + 24 * v8;
      *(_BYTE *)(result + 32) = 0;
      return result;
    }
    if ( !v16 )
      break;
LABEL_5:
    v14 = (v8 - 1) & (v38 + v14);
    ++v38;
  }
  v17 = *((_DWORD *)v15 + 2);
  if ( v17 != -1 )
  {
    if ( v17 == -2 && !v13 )
      v13 = v12 + 24LL * v14;
    goto LABEL_5;
  }
  if ( !v13 )
    v13 = v12 + 24LL * v14;
  v26 = *(_DWORD *)(a2 + 16) + 1;
  *(_QWORD *)a2 = v9 + 1;
  if ( 4 * v26 >= (unsigned int)(3 * v8) )
  {
LABEL_14:
    v39 = result;
    sub_1FE7AA0(a2, 2 * v8);
    v18 = *(_DWORD *)(a2 + 24);
    if ( v18 )
    {
      v19 = v18 - 1;
      v20 = *((_DWORD *)a3 + 2);
      v21 = 0;
      result = v39;
      v23 = 1;
      for ( i = (v18 - 1) & (v20 + ((*a3 >> 9) ^ (*a3 >> 4))); ; i = v19 & v25 )
      {
        v22 = *(_QWORD *)(a2 + 8);
        v13 = v22 + 24LL * i;
        if ( *a3 == *(_QWORD *)v13 && v20 == *(_DWORD *)(v13 + 8) )
          break;
        if ( !*(_QWORD *)v13 )
        {
          v36 = *(_DWORD *)(v13 + 8);
          if ( v36 == -1 )
          {
            if ( v21 )
              v13 = v21;
            v26 = *(_DWORD *)(a2 + 16) + 1;
            goto LABEL_23;
          }
          if ( !v21 && v36 == -2 )
            v21 = v22 + 24LL * i;
        }
        v25 = v23 + i;
        ++v23;
      }
      goto LABEL_34;
    }
LABEL_53:
    ++*(_DWORD *)(a2 + 16);
    BUG();
  }
  if ( (int)v8 - *(_DWORD *)(a2 + 20) - v26 <= (unsigned int)v8 >> 3 )
  {
    v40 = result;
    sub_1FE7AA0(a2, v8);
    v29 = *(_DWORD *)(a2 + 24);
    if ( v29 )
    {
      v30 = v29 - 1;
      v31 = *((_DWORD *)a3 + 2);
      v32 = 1;
      result = v40;
      for ( j = (v29 - 1) & (v31 + ((*a3 >> 9) ^ (*a3 >> 4))); ; j = v30 & v35 )
      {
        v33 = *(_QWORD *)(a2 + 8);
        v13 = v33 + 24LL * j;
        if ( *a3 == *(_QWORD *)v13 && v31 == *(_DWORD *)(v13 + 8) )
          break;
        if ( !*(_QWORD *)v13 )
        {
          v37 = *(_DWORD *)(v13 + 8);
          if ( v37 == -1 )
          {
            if ( v16 )
              v13 = v16;
            v26 = *(_DWORD *)(a2 + 16) + 1;
            goto LABEL_23;
          }
          if ( v37 == -2 && !v16 )
            v16 = v33 + 24LL * j;
        }
        v35 = v32 + j;
        ++v32;
      }
LABEL_34:
      v26 = *(_DWORD *)(a2 + 16) + 1;
      goto LABEL_23;
    }
    goto LABEL_53;
  }
LABEL_23:
  *(_DWORD *)(a2 + 16) = v26;
  if ( *(_QWORD *)v13 || *(_DWORD *)(v13 + 8) != -1 )
    --*(_DWORD *)(a2 + 20);
  v27 = *a3;
  *(_QWORD *)result = a2;
  *(_QWORD *)(result + 16) = v13;
  *(_QWORD *)v13 = v27;
  LODWORD(v27) = *((_DWORD *)a3 + 2);
  *(_BYTE *)(result + 32) = 1;
  *(_DWORD *)(v13 + 8) = v27;
  *(_DWORD *)(v13 + 16) = *a4;
  v28 = *(_QWORD *)a2;
  *(_QWORD *)(result + 24) = *(_QWORD *)(a2 + 8) + 24LL * *(unsigned int *)(a2 + 24);
  *(_QWORD *)(result + 8) = v28;
  return result;
}
