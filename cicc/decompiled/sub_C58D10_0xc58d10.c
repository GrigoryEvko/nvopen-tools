// Function: sub_C58D10
// Address: 0xc58d10
//
__int64 __fastcall sub_C58D10(_QWORD *a1, char *a2, size_t a3, _QWORD *a4)
{
  void **p_src; // rsi
  unsigned int v5; // eax
  __int64 v6; // r9
  __int64 v7; // rbx
  __int64 v8; // r13
  int v9; // edx
  size_t v10; // rbx
  const void *v11; // r15
  size_t v12; // rdi
  size_t v13; // rdx
  char *v14; // rdi
  void (__fastcall *v15)(_QWORD *, void **, char **); // rcx
  unsigned int v16; // r14d
  size_t v18; // rax
  char *v19; // rdx
  void (__fastcall *v20)(_QWORD *, void **, char **); // rcx
  int v21; // r12d
  void **v22; // r12
  __int64 v23; // rdi
  size_t v24; // rbx
  void **v25; // r12
  __int64 v26; // rdi
  size_t v27; // rbx
  int v29; // [rsp+Ch] [rbp-264h]
  int v30; // [rsp+Ch] [rbp-264h]
  __int64 v34; // [rsp+38h] [rbp-238h]
  _QWORD v35[4]; // [rsp+40h] [rbp-230h] BYREF
  __int16 v36; // [rsp+60h] [rbp-210h]
  char *v37; // [rsp+70h] [rbp-200h] BYREF
  size_t v38; // [rsp+78h] [rbp-1F8h]
  __int16 v39; // [rsp+90h] [rbp-1E0h]
  _QWORD v40[2]; // [rsp+A0h] [rbp-1D0h] BYREF
  __int64 v41; // [rsp+B0h] [rbp-1C0h] BYREF
  __int16 v42; // [rsp+C0h] [rbp-1B0h]
  int v43; // [rsp+E8h] [rbp-188h]
  char v44; // [rsp+F8h] [rbp-178h]
  void *src; // [rsp+100h] [rbp-170h] BYREF
  size_t n; // [rsp+108h] [rbp-168h]
  unsigned __int64 v47; // [rsp+110h] [rbp-160h]
  _BYTE v48[136]; // [rsp+118h] [rbp-158h] BYREF
  char *v49; // [rsp+1A0h] [rbp-D0h] BYREF
  size_t v50; // [rsp+1A8h] [rbp-C8h]
  __int64 v51; // [rsp+1B0h] [rbp-C0h]
  char v52; // [rsp+1B8h] [rbp-B8h] BYREF
  __int16 v53; // [rsp+1C0h] [rbp-B0h]

  v49 = a2;
  p_src = 0;
  src = v48;
  n = 0;
  v47 = 128;
  v53 = 261;
  v50 = a3;
  v5 = sub_C81CA0(&v49, 0);
  if ( !(_BYTE)v5 )
  {
    v6 = a1[5];
    v7 = 16LL * a1[6];
    v34 = v6 + v7;
    if ( v6 + v7 != v6 )
    {
      v8 = a1[5];
      while ( 1 )
      {
        v10 = *(_QWORD *)(v8 + 8);
        if ( !v10 )
          goto LABEL_9;
        v11 = *(const void **)v8;
        v12 = 0;
        n = 0;
        if ( v10 > v47 )
        {
          sub_C8D290(&src, v48, v10, 1);
          v12 = n;
        }
        memcpy((char *)src + v12, v11, v10);
        v35[0] = a2;
        v53 = 257;
        v42 = 257;
        v39 = 257;
        v36 = 261;
        n += v10;
        v35[1] = a3;
        sub_C81B70(&src, v35, &v37, v40, &v49);
        sub_C83970(&src, 0);
        v13 = n;
        v49 = &v52;
        v14 = &v52;
        v50 = 0;
        v51 = 128;
        if ( n )
        {
          sub_C4FA10((__int64)&v49, (__int64)&src);
          v14 = v49;
          v13 = v50;
        }
        p_src = (void **)a1[2];
        v15 = (void (__fastcall *)(_QWORD *, void **, char **))*((_QWORD *)*p_src + 5);
        v37 = v14;
        v38 = v13;
        v39 = 261;
        v15(v40, p_src, &v37);
        if ( (v44 & 1) != 0 )
        {
          if ( v49 == &v52 )
            goto LABEL_9;
          _libc_free(v49, p_src);
          v8 += 16;
          if ( v34 == v8 )
            break;
        }
        else
        {
          v9 = v43;
          if ( (__int64 *)v40[0] != &v41 )
          {
            v29 = v43;
            p_src = (void **)(v41 + 1);
            j_j___libc_free_0(v40[0], v41 + 1);
            v9 = v29;
          }
          if ( v49 != &v52 )
          {
            v30 = v9;
            _libc_free(v49, p_src);
            v9 = v30;
          }
          if ( v9 == 2 )
          {
            v25 = (void **)src;
            v26 = 0;
            v27 = n;
            a4[1] = 0;
            if ( v27 > a4[2] )
            {
              p_src = (void **)(a4 + 3);
              sub_C8D290(a4, a4 + 3, v27, 1);
              v26 = a4[1];
            }
            if ( v27 )
            {
              p_src = v25;
              memcpy((void *)(*a4 + v26), v25, v27);
              v26 = a4[1];
            }
            v16 = 1;
            a4[1] = v27 + v26;
            goto LABEL_19;
          }
LABEL_9:
          v8 += 16;
          if ( v34 == v8 )
            break;
        }
      }
    }
LABEL_18:
    v16 = 0;
    goto LABEL_19;
  }
  n = 0;
  v16 = v5;
  sub_C58CA0(&src, a2, &a2[a3]);
  v53 = 261;
  v49 = a2;
  v50 = a3;
  if ( (unsigned __int8)sub_C81F30(&v49, 0) )
  {
    p_src = &src;
    if ( (*(unsigned int (__fastcall **)(_QWORD, void **))(*(_QWORD *)a1[2] + 112LL))(a1[2], &src) )
      goto LABEL_18;
  }
  v18 = n;
  v50 = 0;
  v49 = &v52;
  v19 = &v52;
  v51 = 128;
  if ( n )
  {
    sub_C4FA10((__int64)&v49, (__int64)&src);
    v19 = v49;
    v18 = v50;
  }
  p_src = (void **)a1[2];
  v20 = (void (__fastcall *)(_QWORD *, void **, char **))*((_QWORD *)*p_src + 5);
  v37 = v19;
  v39 = 261;
  v38 = v18;
  v20(v40, p_src, &v37);
  if ( (v44 & 1) != 0 )
  {
    v16 = 0;
    if ( v49 == &v52 )
      goto LABEL_19;
    _libc_free(v49, p_src);
    goto LABEL_18;
  }
  v21 = v43;
  sub_2240A30(v40);
  if ( v49 != &v52 )
    _libc_free(v49, p_src);
  if ( v21 != 2 )
    goto LABEL_18;
  v22 = (void **)src;
  v23 = 0;
  v24 = n;
  a4[1] = 0;
  if ( v24 > a4[2] )
  {
    p_src = (void **)(a4 + 3);
    sub_C8D290(a4, a4 + 3, v24, 1);
    v23 = a4[1];
  }
  if ( v24 )
  {
    p_src = v22;
    memcpy((void *)(*a4 + v23), v22, v24);
    v23 = a4[1];
  }
  a4[1] = v23 + v24;
LABEL_19:
  if ( src != v48 )
    _libc_free(src, p_src);
  return v16;
}
