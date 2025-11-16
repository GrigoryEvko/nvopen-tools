// Function: sub_E6F7B0
// Address: 0xe6f7b0
//
_BYTE *__fastcall sub_E6F7B0(unsigned __int64 *a1, __int64 a2, __int64 a3, __int64 *a4)
{
  __int64 v5; // rcx
  _BYTE *v6; // rdi
  size_t v7; // rsi
  __int64 v8; // rdx
  _BYTE *v9; // rdi
  size_t v10; // rsi
  __int64 v11; // rdx
  _BYTE *v12; // rdi
  size_t v13; // rdx
  size_t v14; // rsi
  __int64 v15; // rax
  __int64 v16; // rdi
  __int64 v17; // rsi
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 *v20; // rsi
  __int64 v21; // r15
  __int64 v22; // r12
  __int64 v23; // rdi
  _BYTE *result; // rax
  size_t v25; // rdx
  size_t v26; // rdx
  _QWORD v27[2]; // [rsp+0h] [rbp-1A0h] BYREF
  _QWORD *v28; // [rsp+10h] [rbp-190h]
  size_t n; // [rsp+18h] [rbp-188h]
  _QWORD src[3]; // [rsp+20h] [rbp-180h] BYREF
  int v31; // [rsp+38h] [rbp-168h]
  _QWORD *v32; // [rsp+40h] [rbp-160h]
  size_t v33; // [rsp+48h] [rbp-158h]
  _QWORD v34[2]; // [rsp+50h] [rbp-150h] BYREF
  _QWORD *v35; // [rsp+60h] [rbp-140h]
  size_t v36; // [rsp+68h] [rbp-138h]
  _QWORD v37[2]; // [rsp+70h] [rbp-130h] BYREF
  __int64 v38; // [rsp+80h] [rbp-120h]
  __int64 v39; // [rsp+88h] [rbp-118h]
  __int64 v40; // [rsp+90h] [rbp-110h]
  __int64 v41; // [rsp+98h] [rbp-108h] BYREF
  unsigned int v42; // [rsp+A0h] [rbp-100h]
  _BYTE v43[248]; // [rsp+A8h] [rbp-F8h] BYREF

  sub_C917B0((__int64)v27, a4, *a1, 0, a2, (__int64)a1, 0, 0, 0, 0);
  v6 = *(_BYTE **)(a3 + 16);
  *(_QWORD *)a3 = v27[0];
  *(_QWORD *)(a3 + 8) = v27[1];
  if ( v28 == src )
  {
    v25 = n;
    if ( n )
    {
      if ( n == 1 )
        *v6 = src[0];
      else
        memcpy(v6, src, n);
      v25 = n;
      v6 = *(_BYTE **)(a3 + 16);
    }
    *(_QWORD *)(a3 + 24) = v25;
    v6[v25] = 0;
    v6 = v28;
  }
  else
  {
    v5 = src[0];
    v7 = n;
    if ( v6 == (_BYTE *)(a3 + 32) )
    {
      *(_QWORD *)(a3 + 16) = v28;
      *(_QWORD *)(a3 + 24) = v7;
      *(_QWORD *)(a3 + 32) = v5;
    }
    else
    {
      v8 = *(_QWORD *)(a3 + 32);
      *(_QWORD *)(a3 + 16) = v28;
      *(_QWORD *)(a3 + 24) = v7;
      *(_QWORD *)(a3 + 32) = v5;
      if ( v6 )
      {
        v28 = v6;
        src[0] = v8;
        goto LABEL_5;
      }
    }
    v28 = src;
    v6 = src;
  }
LABEL_5:
  n = 0;
  *v6 = 0;
  v9 = *(_BYTE **)(a3 + 64);
  *(_QWORD *)(a3 + 48) = src[2];
  *(_DWORD *)(a3 + 56) = v31;
  if ( v32 == v34 )
  {
    v26 = v33;
    if ( v33 )
    {
      if ( v33 == 1 )
        *v9 = v34[0];
      else
        memcpy(v9, v34, v33);
      v26 = v33;
      v9 = *(_BYTE **)(a3 + 64);
    }
    *(_QWORD *)(a3 + 72) = v26;
    v9[v26] = 0;
    v9 = v32;
  }
  else
  {
    v10 = v33;
    v5 = v34[0];
    if ( v9 == (_BYTE *)(a3 + 80) )
    {
      *(_QWORD *)(a3 + 64) = v32;
      *(_QWORD *)(a3 + 72) = v10;
      *(_QWORD *)(a3 + 80) = v5;
    }
    else
    {
      v11 = *(_QWORD *)(a3 + 80);
      *(_QWORD *)(a3 + 64) = v32;
      *(_QWORD *)(a3 + 72) = v10;
      *(_QWORD *)(a3 + 80) = v5;
      if ( v9 )
      {
        v32 = v9;
        v34[0] = v11;
        goto LABEL_9;
      }
    }
    v32 = v34;
    v9 = v34;
  }
LABEL_9:
  v33 = 0;
  *v9 = 0;
  v12 = *(_BYTE **)(a3 + 96);
  if ( v35 == v37 )
  {
    v13 = v36;
    if ( v36 )
    {
      if ( v36 == 1 )
        *v12 = v37[0];
      else
        memcpy(v12, v37, v36);
      v13 = v36;
      v12 = *(_BYTE **)(a3 + 96);
    }
    *(_QWORD *)(a3 + 104) = v13;
    v12[v13] = 0;
    v12 = v35;
  }
  else
  {
    v13 = a3 + 112;
    v14 = v36;
    v5 = v37[0];
    if ( v12 == (_BYTE *)(a3 + 112) )
    {
      *(_QWORD *)(a3 + 96) = v35;
      *(_QWORD *)(a3 + 104) = v14;
      *(_QWORD *)(a3 + 112) = v5;
    }
    else
    {
      v13 = *(_QWORD *)(a3 + 112);
      *(_QWORD *)(a3 + 96) = v35;
      *(_QWORD *)(a3 + 104) = v14;
      *(_QWORD *)(a3 + 112) = v5;
      if ( v12 )
      {
        v35 = v12;
        v37[0] = v13;
        goto LABEL_13;
      }
    }
    v35 = v37;
    v12 = v37;
  }
LABEL_13:
  v36 = 0;
  *v12 = 0;
  v15 = v38;
  v16 = *(_QWORD *)(a3 + 128);
  v17 = *(_QWORD *)(a3 + 144);
  v38 = 0;
  *(_QWORD *)(a3 + 128) = v15;
  v18 = v39;
  v39 = 0;
  *(_QWORD *)(a3 + 136) = v18;
  v19 = v40;
  v40 = 0;
  *(_QWORD *)(a3 + 144) = v19;
  if ( v16 )
    j_j___libc_free_0(v16, v17 - v16);
  v20 = &v41;
  sub_E6EDB0(a3 + 152, (__int64)&v41, v13, v5);
  v21 = v41;
  v22 = v41 + 48LL * v42;
  if ( v41 != v22 )
  {
    do
    {
      v22 -= 48;
      v23 = *(_QWORD *)(v22 + 16);
      if ( v23 != v22 + 32 )
      {
        v20 = (__int64 *)(*(_QWORD *)(v22 + 32) + 1LL);
        j_j___libc_free_0(v23, v20);
      }
    }
    while ( v21 != v22 );
    v22 = v41;
  }
  result = v43;
  if ( (_BYTE *)v22 != v43 )
    result = (_BYTE *)_libc_free(v22, v20);
  if ( v38 )
    result = (_BYTE *)j_j___libc_free_0(v38, v40 - v38);
  if ( v35 != v37 )
    result = (_BYTE *)j_j___libc_free_0(v35, v37[0] + 1LL);
  if ( v32 != v34 )
    result = (_BYTE *)j_j___libc_free_0(v32, v34[0] + 1LL);
  if ( v28 != src )
    return (_BYTE *)j_j___libc_free_0(v28, src[0] + 1LL);
  return result;
}
