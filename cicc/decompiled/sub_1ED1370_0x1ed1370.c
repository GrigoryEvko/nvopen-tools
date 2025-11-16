// Function: sub_1ED1370
// Address: 0x1ed1370
//
__int64 __fastcall sub_1ED1370(_QWORD *a1, unsigned int a2)
{
  __int64 v4; // rdx
  unsigned int **v5; // rsi
  unsigned int v6; // r14d
  __int64 v7; // r15
  __int64 v8; // rax
  unsigned int v9; // r13d
  __int64 v10; // rbx
  unsigned int *v11; // r8
  unsigned int v12; // ecx
  unsigned __int64 v13; // rsi
  unsigned int v14; // edi
  float *v15; // rcx
  float *v16; // rsi
  __int64 v17; // rdx
  float v18; // xmm1_4
  __int64 i; // rax
  float v20; // xmm0_4
  __int64 v21; // rdx
  void *v22; // rdi
  unsigned int v23; // eax
  _QWORD *v24; // rbx
  volatile signed __int32 *v25; // rdi
  volatile signed __int32 *v26; // r15
  __int64 result; // rax
  unsigned int v28; // r15d
  __int64 v29; // rsi
  float *v30; // rdx
  float v31; // xmm1_4
  float *v32; // rax
  __int64 v33; // r9
  unsigned int v34; // edx
  __int64 v35; // r11
  float *v36; // r10
  const void **v37; // [rsp+0h] [rbp-90h]
  unsigned int v38; // [rsp+Ch] [rbp-84h]
  unsigned int *v39; // [rsp+10h] [rbp-80h]
  __int64 v40; // [rsp+18h] [rbp-78h]
  unsigned int v41; // [rsp+20h] [rbp-70h]
  void *dest; // [rsp+28h] [rbp-68h] BYREF
  unsigned int v43; // [rsp+30h] [rbp-60h]
  void *v44; // [rsp+38h] [rbp-58h] BYREF
  __int64 v45; // [rsp+40h] [rbp-50h] BYREF
  volatile signed __int32 *v46; // [rsp+48h] [rbp-48h]
  unsigned int v47; // [rsp+50h] [rbp-40h] BYREF
  void *v48; // [rsp+58h] [rbp-38h]

  v4 = a1[20];
  v5 = (unsigned int **)(v4 + 88LL * a2);
  v6 = *v5[8];
  v7 = 48LL * v6;
  v8 = v7 + a1[26];
  v9 = *(_DWORD *)(v8 + 20);
  if ( a2 == v9 )
    v9 = *(_DWORD *)(v8 + 24);
  v10 = *(_QWORD *)v8;
  v38 = a2;
  v39 = *v5;
  v40 = 88LL * v9;
  v37 = *(const void ***)(v4 + v40);
  v41 = *(_DWORD *)v37;
  sub_1ECC890(&dest, *(unsigned int *)v37);
  v11 = v39;
  v12 = v38;
  v13 = v41;
  if ( 4LL * v41 )
  {
    memmove(dest, v37[1], 4LL * v41);
    v13 = v41;
    v12 = v38;
    v11 = v39;
  }
  if ( v12 == *(_DWORD *)(a1[26] + v7 + 20) )
  {
    if ( (_DWORD)v13 )
    {
      v28 = 0;
      do
      {
        v29 = *(_QWORD *)(v10 + 8);
        v30 = (float *)*((_QWORD *)v11 + 1);
        v31 = *(float *)(v29 + 4LL * v28) + *v30;
        if ( *v11 > 1 )
        {
          v32 = v30 + 1;
          v33 = (__int64)&v30[*v11];
          v34 = *(_DWORD *)(v10 + 4);
          do
          {
            v35 = v34;
            ++v32;
            v34 += *(_DWORD *)(v10 + 4);
            v31 = fminf(*(float *)(v29 + 4 * (v28 + v35)) + *(v32 - 1), v31);
          }
          while ( (float *)v33 != v32 );
        }
        v36 = (float *)((char *)dest + 4 * v28++);
        *v36 = v31 + *v36;
        v13 = v41;
      }
      while ( v28 < v41 );
    }
  }
  else if ( (_DWORD)v13 )
  {
    v14 = 0;
    do
    {
      v15 = (float *)*((_QWORD *)v11 + 1);
      v16 = (float *)(*(_QWORD *)(v10 + 8) + 4LL * *(_DWORD *)(v10 + 4) * v14);
      v17 = *v11;
      v18 = *v16 + *v15;
      if ( (unsigned int)v17 > 1 )
      {
        for ( i = 1; i != v17; ++i )
        {
          v20 = v16[i] + v15[i];
          v18 = fminf(v20, v18);
        }
      }
      v21 = v14++;
      *((float *)dest + v21) = v18 + *((float *)dest + v21);
      v13 = v41;
    }
    while ( v14 < v41 );
  }
  v43 = v13;
  sub_1ECC890(&v44, v13);
  v22 = v44;
  v23 = v43;
  if ( 4LL * v43 )
  {
    memmove(v44, dest, 4LL * v43);
    v23 = v43;
    v22 = v44;
  }
  v48 = v22;
  v44 = 0;
  v43 = 0;
  v47 = v23;
  sub_1ED0750(&v45, (__int64)(a1 + 11), &v47);
  if ( v48 )
    j_j___libc_free_0_0(v48);
  v24 = (_QWORD *)(a1[20] + v40);
  *v24 = v45;
  v25 = (volatile signed __int32 *)v24[1];
  v26 = v46;
  if ( v46 != v25 )
  {
    if ( v46 )
    {
      if ( &_pthread_key_create )
        _InterlockedAdd(v46 + 2, 1u);
      else
        ++*((_DWORD *)v46 + 2);
      v25 = (volatile signed __int32 *)v24[1];
    }
    if ( v25 )
      sub_A191D0(v25);
    v24[1] = v26;
    v25 = v46;
  }
  if ( v25 )
    sub_A191D0(v25);
  if ( v44 )
    j_j___libc_free_0_0(v44);
  result = sub_1ECDA00(a1, v6, v9);
  if ( dest )
    return j_j___libc_free_0_0(dest);
  return result;
}
