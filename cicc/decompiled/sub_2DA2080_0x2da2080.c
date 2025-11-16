// Function: sub_2DA2080
// Address: 0x2da2080
//
_QWORD *__fastcall sub_2DA2080(_QWORD *a1, __int64 a2, unsigned __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v7; // r13
  _QWORD *v8; // r12
  __int64 v9; // rbx
  __int64 v10; // rax
  _QWORD *v11; // r14
  int v12; // ecx
  _QWORD *v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rsi
  unsigned int v17; // r8d
  __int64 v18; // r11
  __int64 v19; // r14
  int v20; // r9d
  __int64 *v21; // rcx
  unsigned int i; // edx
  __int64 *v23; // rdi
  __int64 v24; // r12
  unsigned int v25; // edx
  int v26; // esi
  int v27; // edx
  __int64 v28; // rax
  _QWORD *v29; // rbx
  volatile signed __int32 *v30; // rdi
  volatile signed __int32 *v31; // r12
  __int64 v32; // rax
  _QWORD *v34; // rax
  __int64 *v35; // rcx
  __int64 v36; // rsi
  unsigned __int64 v37; // r12
  _QWORD *v38; // rsi
  __int64 v39; // rdx
  __int64 v40; // rdx
  __int64 v41; // rcx
  __int64 v42; // r8
  __int64 v43; // rbx
  __int64 v44; // r12
  volatile signed __int32 *v45; // r13
  signed __int32 v46; // edx
  signed __int32 v47; // edx
  int v48; // eax
  int v49; // eax
  _QWORD *v50; // [rsp+8h] [rbp-78h]
  __int64 v51; // [rsp+10h] [rbp-70h]
  __int64 v52; // [rsp+18h] [rbp-68h]
  int v53; // [rsp+18h] [rbp-68h]
  char v54; // [rsp+27h] [rbp-59h]
  __int64 v55; // [rsp+28h] [rbp-58h]
  __int64 v56; // [rsp+28h] [rbp-58h]
  __int64 *v57; // [rsp+38h] [rbp-48h] BYREF
  __int64 v58; // [rsp+40h] [rbp-40h] BYREF
  __int64 v59; // [rsp+48h] [rbp-38h]

  v7 = (_QWORD *)a3;
  v8 = (_QWORD *)a3;
  v9 = a2;
  v10 = *(unsigned int *)(a2 + 24);
  v11 = *(_QWORD **)(a2 + 16);
  v12 = *(_DWORD *)(a2 + 24);
  if ( v10 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 28) )
  {
    if ( (unsigned __int64)v11 <= a3 && a3 < (unsigned __int64)&v11[2 * v10] )
    {
      v54 = 1;
      v56 = (__int64)(a3 - (_QWORD)v11) >> 4;
    }
    else
    {
      v56 = -1;
      v54 = 0;
    }
    v52 = a2 + 32;
    v34 = (_QWORD *)sub_C8D7D0(a2 + 16, a2 + 32, v10 + 1, 0x10u, (unsigned __int64 *)&v58, a6);
    v35 = *(__int64 **)(a2 + 16);
    v11 = v34;
    v36 = 2LL * *(unsigned int *)(a2 + 24);
    v37 = (unsigned __int64)&v35[v36];
    if ( v35 != &v35[v36] )
    {
      v38 = &v34[v36];
      do
      {
        if ( v34 )
        {
          v39 = *v35;
          v34[1] = 0;
          *v34 = v39;
          v40 = v35[1];
          v35[1] = 0;
          v34[1] = v40;
          *v35 = 0;
        }
        v34 += 2;
        v35 += 2;
      }
      while ( v34 != v38 );
      v41 = *(_QWORD *)(v9 + 16);
      v42 = 16LL * *(unsigned int *)(v9 + 24);
      v37 = v41 + v42;
      if ( v41 != v41 + v42 )
      {
        v51 = v9;
        v43 = v41 + v42;
        v44 = v41;
        v50 = v7;
        do
        {
          v45 = *(volatile signed __int32 **)(v43 - 8);
          v43 -= 16;
          if ( v45 )
          {
            if ( &_pthread_key_create )
            {
              v46 = _InterlockedExchangeAdd(v45 + 2, 0xFFFFFFFF);
            }
            else
            {
              v46 = *((_DWORD *)v45 + 2);
              *((_DWORD *)v45 + 2) = v46 - 1;
            }
            if ( v46 == 1 )
            {
              (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v45 + 16LL))(v45);
              if ( &_pthread_key_create )
              {
                v47 = _InterlockedExchangeAdd(v45 + 3, 0xFFFFFFFF);
              }
              else
              {
                v47 = *((_DWORD *)v45 + 3);
                *((_DWORD *)v45 + 3) = v47 - 1;
              }
              if ( v47 == 1 )
                (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v45 + 24LL))(v45);
            }
          }
        }
        while ( v43 != v44 );
        v9 = v51;
        v7 = v50;
        v37 = *(_QWORD *)(v51 + 16);
      }
    }
    v48 = v58;
    if ( v52 != v37 )
    {
      v53 = v58;
      _libc_free(v37);
      v48 = v53;
    }
    *(_DWORD *)(v9 + 28) = v48;
    v10 = *(unsigned int *)(v9 + 24);
    *(_QWORD *)(v9 + 16) = v11;
    v8 = &v11[2 * v56];
    v12 = v10;
    if ( !v54 )
      v8 = v7;
  }
  v13 = &v11[2 * v10];
  if ( v13 )
  {
    *v13 = *v8;
    v14 = v8[1];
    v13[1] = v14;
    if ( !v14 )
    {
LABEL_6:
      v12 = *(_DWORD *)(v9 + 24);
      goto LABEL_7;
    }
    if ( !&_pthread_key_create )
    {
      ++*(_DWORD *)(v14 + 8);
      goto LABEL_6;
    }
    _InterlockedAdd((volatile signed __int32 *)(v14 + 8), 1u);
    v12 = *(_DWORD *)(v9 + 24);
  }
LABEL_7:
  *(_DWORD *)(v9 + 24) = v12 + 1;
  v15 = *v7;
  if ( !*(_QWORD *)(*v7 + 8LL) )
    goto LABEL_32;
  v16 = *(_QWORD *)(v15 + 8);
  v17 = *(_DWORD *)(v9 + 104);
  v55 = v9 + 80;
  v58 = v16;
  v18 = *(_QWORD *)(v15 + 16);
  v59 = v18;
  if ( !v17 )
  {
    ++*(_QWORD *)(v9 + 80);
    v57 = 0;
    goto LABEL_18;
  }
  v19 = *(_QWORD *)(v9 + 88);
  v20 = 1;
  v21 = 0;
  for ( i = (v17 - 1)
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4)
              | ((unsigned __int64)(((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4)))); ; i = (v17 - 1) & v25 )
  {
    v23 = (__int64 *)(v19 + 32LL * i);
    v24 = *v23;
    if ( v16 == *v23 && v18 == v23[1] )
    {
      v29 = v23 + 2;
      v30 = (volatile signed __int32 *)v23[3];
      goto LABEL_23;
    }
    if ( v24 == -4096 )
      break;
    if ( v24 == -8192 && v23[1] == -8192 && !v21 )
      v21 = (__int64 *)(v19 + 32LL * i);
LABEL_16:
    v25 = v20 + i;
    ++v20;
  }
  if ( v23[1] != -4096 )
    goto LABEL_16;
  v49 = *(_DWORD *)(v9 + 96);
  if ( !v21 )
    v21 = (__int64 *)(v19 + 32LL * i);
  ++*(_QWORD *)(v9 + 80);
  v27 = v49 + 1;
  v57 = v21;
  if ( 4 * (v49 + 1) >= 3 * v17 )
  {
LABEL_18:
    v26 = 2 * v17;
    goto LABEL_19;
  }
  if ( v17 - *(_DWORD *)(v9 + 100) - v27 <= v17 >> 3 )
  {
    v26 = v17;
LABEL_19:
    sub_2D9FC50(v55, v26);
    sub_2D9F6D0(v55, &v58, &v57);
    v16 = v58;
    v21 = v57;
    v27 = *(_DWORD *)(v9 + 96) + 1;
  }
  *(_DWORD *)(v9 + 96) = v27;
  if ( *v21 != -4096 || v21[1] != -4096 )
    --*(_DWORD *)(v9 + 100);
  *v21 = v16;
  v28 = v59;
  v29 = v21 + 2;
  v30 = 0;
  v21[2] = 0;
  v21[1] = v28;
  v21[3] = 0;
  v15 = *v7;
LABEL_23:
  *v29 = v15;
  v31 = (volatile signed __int32 *)v7[1];
  if ( v31 != v30 )
  {
    if ( v31 )
    {
      if ( &_pthread_key_create )
        _InterlockedAdd(v31 + 2, 1u);
      else
        ++*((_DWORD *)v31 + 2);
      v30 = (volatile signed __int32 *)v29[1];
    }
    if ( v30 )
      sub_A191D0(v30);
    v29[1] = v31;
  }
  v15 = *v7;
LABEL_32:
  a1[1] = 0;
  *a1 = v15;
  v32 = v7[1];
  *v7 = 0;
  v7[1] = 0;
  a1[1] = v32;
  return a1;
}
