// Function: sub_35BF310
// Address: 0x35bf310
//
void __fastcall sub_35BF310(_QWORD *a1, unsigned int a2)
{
  __int64 v3; // rax
  __int64 v4; // rsi
  unsigned int **v5; // rax
  __int64 v6; // r13
  __int64 v7; // rdx
  unsigned int *v8; // r15
  unsigned __int64 v9; // r14
  void *v10; // rax
  __int64 v11; // r10
  unsigned int *v12; // r9
  __int64 v13; // rcx
  void *v14; // r12
  __int64 v15; // rbx
  float *v16; // rsi
  float *v17; // rdi
  __int64 v18; // rdx
  float v19; // xmm1_4
  __int64 i; // rax
  float v21; // xmm0_4
  void *v22; // rax
  void *v23; // rcx
  _QWORD *v24; // r13
  volatile signed __int32 *v25; // rdi
  volatile signed __int32 *v26; // rbx
  __int64 v27; // r11
  __int64 v28; // rdi
  float *v29; // rdx
  float v30; // xmm1_4
  float *v31; // rax
  __int64 v32; // rbx
  unsigned int v33; // esi
  __int64 v34; // rdx
  __int64 v35; // [rsp+8h] [rbp-88h]
  __int64 v36; // [rsp+8h] [rbp-88h]
  __int64 v37; // [rsp+8h] [rbp-88h]
  unsigned int *v38; // [rsp+10h] [rbp-80h]
  unsigned int *v39; // [rsp+10h] [rbp-80h]
  __int64 v40; // [rsp+18h] [rbp-78h]
  __int64 v41; // [rsp+18h] [rbp-78h]
  __int64 v42; // [rsp+18h] [rbp-78h]
  unsigned int v43; // [rsp+2Ch] [rbp-64h]
  unsigned int v44; // [rsp+30h] [rbp-60h]
  unsigned int v45; // [rsp+34h] [rbp-5Ch]
  __int64 v47; // [rsp+40h] [rbp-50h] BYREF
  volatile signed __int32 *v48; // [rsp+48h] [rbp-48h]
  unsigned int v49; // [rsp+50h] [rbp-40h] BYREF
  unsigned __int64 v50; // [rsp+58h] [rbp-38h]

  v3 = 96LL * a2;
  v4 = a1[20];
  v5 = (unsigned int **)(v4 + v3);
  v43 = *v5[9];
  v6 = 48LL * v43;
  v7 = v6 + a1[26];
  v45 = *(_DWORD *)(v7 + 20);
  if ( a2 == v45 )
    v45 = *(_DWORD *)(v7 + 24);
  v38 = *v5;
  v35 = *(_QWORD *)v7;
  v8 = *(unsigned int **)(v4 + 96LL * v45);
  v9 = 4LL * *v8;
  v44 = *v8;
  v40 = *v8;
  v10 = (void *)sub_2207820(v9);
  v11 = v40;
  v12 = v38;
  v13 = v35;
  v14 = v10;
  if ( v10 && v40 )
  {
    v36 = v40;
    v41 = v13;
    memset(v10, 0, v9);
    v13 = v41;
    v12 = v38;
    v11 = v36;
  }
  if ( v9 )
  {
    v39 = v12;
    v37 = v11;
    v42 = v13;
    memmove(v14, *((const void **)v8 + 1), v9);
    v11 = v37;
    v12 = v39;
    v13 = v42;
  }
  if ( a2 == *(_DWORD *)(a1[26] + v6 + 20) )
  {
    if ( v44 )
    {
      v27 = 0;
      do
      {
        v28 = *(_QWORD *)(v13 + 8);
        v29 = (float *)*((_QWORD *)v12 + 1);
        v30 = *(float *)(v28 + 4 * v27) + *v29;
        if ( *v12 > 1 )
        {
          v31 = v29 + 1;
          v32 = (__int64)&v29[*v12];
          v33 = *(_DWORD *)(v13 + 4);
          do
          {
            v34 = v33;
            ++v31;
            v33 += *(_DWORD *)(v13 + 4);
            v30 = fminf(*(float *)(v28 + 4 * (v27 + v34)) + *(v31 - 1), v30);
          }
          while ( v31 != (float *)v32 );
        }
        *((float *)v14 + v27) = v30 + *((float *)v14 + v27);
        ++v27;
      }
      while ( v11 != v27 );
      goto LABEL_15;
    }
LABEL_41:
    v23 = (void *)sub_2207820(v9);
    goto LABEL_17;
  }
  if ( !v44 )
    goto LABEL_41;
  v15 = 0;
  do
  {
    v16 = (float *)*((_QWORD *)v12 + 1);
    v17 = (float *)(*(_QWORD *)(v13 + 8) + 4LL * (unsigned int)(*(_DWORD *)(v13 + 4) * v15));
    v18 = *v12;
    v19 = *v17 + *v16;
    if ( (unsigned int)v18 > 1 )
    {
      for ( i = 1; i != v18; ++i )
      {
        v21 = v17[i] + v16[i];
        v19 = fminf(v21, v19);
      }
    }
    *((float *)v14 + v15) = v19 + *((float *)v14 + v15);
    ++v15;
  }
  while ( v11 != v15 );
LABEL_15:
  v22 = (void *)sub_2207820(v9);
  v23 = v22;
  if ( v22 )
    v23 = memset(v22, 0, v9);
LABEL_17:
  if ( v9 )
    v23 = memcpy(v23, v14, v9);
  v50 = (unsigned __int64)v23;
  v49 = v44;
  sub_35BE780(&v47, (__int64)(a1 + 11), &v49);
  if ( v50 )
    j_j___libc_free_0_0(v50);
  v24 = (_QWORD *)(a1[20] + 96LL * v45);
  v25 = (volatile signed __int32 *)v24[1];
  *v24 = v47;
  v26 = v48;
  if ( v48 != v25 )
  {
    if ( v48 )
    {
      if ( &_pthread_key_create )
        _InterlockedAdd(v48 + 2, 1u);
      else
        ++*((_DWORD *)v48 + 2);
      v25 = (volatile signed __int32 *)v24[1];
    }
    if ( v25 )
      sub_A191D0(v25);
    v24[1] = v26;
    v25 = v48;
  }
  if ( v25 )
    sub_A191D0(v25);
  sub_35BAE80(a1, v43, v45);
  if ( v14 )
    j_j___libc_free_0_0((unsigned __int64)v14);
}
