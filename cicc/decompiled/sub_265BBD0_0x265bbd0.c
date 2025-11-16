// Function: sub_265BBD0
// Address: 0x265bbd0
//
__int64 __fastcall sub_265BBD0(__int64 a1, __int64 *a2)
{
  __int64 v4; // rdx
  unsigned int v5; // esi
  __int64 v6; // rdi
  int v7; // r10d
  __int64 v8; // r12
  unsigned int v9; // ecx
  __int64 v10; // rax
  __int64 v11; // r9
  __int64 v12; // rax
  int v14; // eax
  int v15; // ecx
  __int64 *v16; // r15
  __int64 v17; // rax
  unsigned __int64 v18; // rcx
  unsigned int v19; // edx
  __int64 v20; // r14
  __int64 *v21; // r9
  __int64 *v22; // rcx
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  unsigned __int64 v26; // rdi
  unsigned __int64 v27; // rdx
  _QWORD *v28; // rax
  __int64 v29; // rsi
  unsigned __int64 v30; // r13
  _QWORD *v31; // rsi
  _QWORD *v32; // rdx
  __int64 v33; // rax
  __int64 v34; // r8
  unsigned __int64 v35; // rdi
  int v36; // eax
  char v37; // [rsp+7h] [rbp-89h]
  __int64 v38; // [rsp+8h] [rbp-88h]
  __int64 v39; // [rsp+18h] [rbp-78h]
  int v40; // [rsp+18h] [rbp-78h]
  unsigned __int64 v41; // [rsp+28h] [rbp-68h] BYREF
  __int64 v42; // [rsp+30h] [rbp-60h] BYREF
  int v43; // [rsp+38h] [rbp-58h]
  __int64 v44; // [rsp+40h] [rbp-50h] BYREF
  unsigned __int64 v45; // [rsp+48h] [rbp-48h]
  __int64 v46; // [rsp+50h] [rbp-40h]
  __int64 v47; // [rsp+58h] [rbp-38h]

  v4 = *a2;
  v5 = *(_DWORD *)(a1 + 24);
  v43 = 0;
  v42 = v4;
  if ( !v5 )
  {
    ++*(_QWORD *)a1;
    v44 = 0;
LABEL_25:
    v16 = &v44;
    sub_9E07A0(a1, 2 * v5);
LABEL_26:
    sub_264B350(a1, &v42, &v44);
    v4 = v42;
    v8 = v44;
    v15 = *(_DWORD *)(a1 + 16) + 1;
    goto LABEL_15;
  }
  v6 = *(_QWORD *)(a1 + 8);
  v7 = 1;
  v8 = 0;
  v9 = (v5 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
  v10 = v6 + 16LL * v9;
  v11 = *(_QWORD *)v10;
  if ( v4 == *(_QWORD *)v10 )
  {
LABEL_3:
    v12 = *(unsigned int *)(v10 + 8);
    return *(_QWORD *)(a1 + 32) + 32 * v12 + 8;
  }
  while ( v11 != -4096 )
  {
    if ( v11 == -8192 && !v8 )
      v8 = v10;
    v9 = (v5 - 1) & (v7 + v9);
    v10 = v6 + 16LL * v9;
    v11 = *(_QWORD *)v10;
    if ( v4 == *(_QWORD *)v10 )
      goto LABEL_3;
    ++v7;
  }
  if ( !v8 )
    v8 = v10;
  v14 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v15 = v14 + 1;
  v44 = v8;
  if ( 4 * (v14 + 1) >= 3 * v5 )
    goto LABEL_25;
  v16 = &v44;
  if ( v5 - *(_DWORD *)(a1 + 20) - v15 <= v5 >> 3 )
  {
    sub_9E07A0(a1, v5);
    goto LABEL_26;
  }
LABEL_15:
  *(_DWORD *)(a1 + 16) = v15;
  if ( *(_QWORD *)v8 != -4096 )
    --*(_DWORD *)(a1 + 20);
  *(_QWORD *)v8 = v4;
  *(_DWORD *)(v8 + 8) = v43;
  v17 = *a2;
  v18 = *(unsigned int *)(a1 + 44);
  v45 = 0;
  v44 = v17;
  v12 = *(unsigned int *)(a1 + 40);
  v46 = 0;
  v47 = 0;
  v19 = v12;
  if ( v12 + 1 > v18 )
  {
    v27 = *(_QWORD *)(a1 + 32);
    if ( v27 > (unsigned __int64)&v44 || (unsigned __int64)&v44 >= v27 + 32 * v12 )
    {
      v38 = -1;
      v37 = 0;
    }
    else
    {
      v37 = 1;
      v38 = (__int64)((__int64)&v44 - v27) >> 5;
    }
    v20 = sub_C8D7D0(a1 + 32, a1 + 48, v12 + 1, 0x20u, &v41, v11);
    v28 = *(_QWORD **)(a1 + 32);
    v29 = 4LL * *(unsigned int *)(a1 + 40);
    v30 = (unsigned __int64)&v28[v29];
    if ( v28 != &v28[v29] )
    {
      v31 = (_QWORD *)(v20 + v29 * 8);
      v32 = (_QWORD *)v20;
      do
      {
        if ( v32 )
        {
          *v32 = *v28;
          v32[1] = v28[1];
          v32[2] = v28[2];
          v32[3] = v28[3];
          v28[3] = 0;
          v28[2] = 0;
          v28[1] = 0;
        }
        v32 += 4;
        v28 += 4;
      }
      while ( v32 != v31 );
      v33 = *(_QWORD *)(a1 + 32);
      v34 = 32LL * *(unsigned int *)(a1 + 40);
      v30 = v33 + v34;
      if ( v33 + v34 != v33 )
      {
        do
        {
          v35 = *(_QWORD *)(v30 - 24);
          v30 -= 32LL;
          if ( v35 )
          {
            v39 = v33;
            j_j___libc_free_0(v35);
            v33 = v39;
          }
        }
        while ( v30 != v33 );
        v30 = *(_QWORD *)(a1 + 32);
      }
    }
    v36 = v41;
    if ( a1 + 48 != v30 )
    {
      v40 = v41;
      _libc_free(v30);
      v36 = v40;
    }
    *(_DWORD *)(a1 + 44) = v36;
    v12 = *(unsigned int *)(a1 + 40);
    *(_QWORD *)(a1 + 32) = v20;
    v19 = v12;
    if ( v37 )
      v16 = (__int64 *)(v20 + 32 * v38);
    v21 = v16;
  }
  else
  {
    v20 = *(_QWORD *)(a1 + 32);
    v21 = &v44;
  }
  v22 = (__int64 *)(32 * v12 + v20);
  if ( v22 )
  {
    *v22 = *v21;
    v23 = v21[1];
    v21[1] = 0;
    v22[1] = v23;
    v24 = v21[2];
    v21[2] = 0;
    v22[2] = v24;
    v25 = v21[3];
    v21[3] = 0;
    v22[3] = v25;
    v19 = *(_DWORD *)(a1 + 40);
    v26 = v45;
    *(_DWORD *)(a1 + 40) = v19 + 1;
    v12 = v19;
    if ( v26 )
    {
      j_j___libc_free_0(v26);
      v12 = (unsigned int)(*(_DWORD *)(a1 + 40) - 1);
      v19 = *(_DWORD *)(a1 + 40) - 1;
    }
  }
  else
  {
    *(_DWORD *)(a1 + 40) = v19 + 1;
  }
  *(_DWORD *)(v8 + 8) = v19;
  return *(_QWORD *)(a1 + 32) + 32 * v12 + 8;
}
