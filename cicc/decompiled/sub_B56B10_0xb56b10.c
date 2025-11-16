// Function: sub_B56B10
// Address: 0xb56b10
//
unsigned __int8 *__fastcall sub_B56B10(__int64 a1, int a2, char *a3, __int64 a4, __int64 a5)
{
  unsigned __int8 *v5; // r14
  bool v7; // sf
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rbx
  __int64 v11; // rbx
  __int64 v12; // r15
  __int64 v13; // rbx
  __int64 v14; // rax
  __int64 v16; // rdx
  char *v17; // rsi
  int v18; // eax
  __int64 v19; // rcx
  __int64 *v20; // r12
  _BYTE *v21; // r10
  size_t v22; // r8
  _BYTE *v23; // rdi
  char *v24; // rdx
  __int64 v25; // rax
  __int64 v26; // rcx
  _BYTE *v27; // rax
  _BYTE *v28; // rsi
  size_t v29; // r13
  __int64 v30; // rax
  char *v31; // r13
  char *v32; // r12
  __int64 v33; // rdi
  __int64 v34; // rax
  __int64 v35; // r13
  int v36; // r12d
  int v37; // r12d
  size_t n; // [rsp+0h] [rbp-B0h]
  void *src; // [rsp+8h] [rbp-A8h]
  __int64 v40; // [rsp+10h] [rbp-A0h]
  __int64 v41; // [rsp+18h] [rbp-98h]
  size_t v42; // [rsp+28h] [rbp-88h] BYREF
  char *v43; // [rsp+30h] [rbp-80h] BYREF
  __int64 v44; // [rsp+38h] [rbp-78h]
  _BYTE v45[112]; // [rsp+40h] [rbp-70h] BYREF

  v5 = (unsigned __int8 *)a1;
  v7 = *(char *)(a1 + 7) < 0;
  v41 = a4;
  v40 = a5;
  if ( !v7 )
    goto LABEL_11;
  v8 = sub_BD2BC0(a1);
  v10 = v8 + v9;
  if ( *(char *)(a1 + 7) < 0 )
    v10 -= sub_BD2BC0(a1);
  v11 = v10 >> 4;
  if ( !(_DWORD)v11 )
  {
LABEL_11:
    v44 = 0x100000000LL;
    v43 = v45;
    sub_B56970(a1, (__int64)&v43);
    v16 = (unsigned int)v44;
    v17 = v43;
    v18 = v44;
    if ( (unsigned __int64)(unsigned int)v44 + 1 > HIDWORD(v44) )
    {
      if ( v43 > a3 || a3 >= &v43[56 * (unsigned int)v44] )
      {
        src = (void *)sub_C8D7D0(&v43, v45, (unsigned int)v44 + 1LL, 56, &v42);
        sub_B56820((__int64)&v43, (__m128i *)src);
        v37 = v42;
        v17 = (char *)src;
        if ( v43 != v45 )
        {
          _libc_free(v43, src);
          v17 = (char *)src;
        }
        v16 = (unsigned int)v44;
        v43 = v17;
        HIDWORD(v44) = v37;
        v18 = v44;
      }
      else
      {
        v35 = a3 - v43;
        src = (void *)sub_C8D7D0(&v43, v45, (unsigned int)v44 + 1LL, 56, &v42);
        sub_B56820((__int64)&v43, (__m128i *)src);
        v36 = v42;
        v17 = (char *)src;
        if ( v43 != v45 )
        {
          _libc_free(v43, src);
          v17 = (char *)src;
        }
        v16 = (unsigned int)v44;
        v43 = v17;
        a3 = &v17[v35];
        HIDWORD(v44) = v36;
        v18 = v44;
      }
    }
    v19 = 7 * v16;
    v20 = (__int64 *)&v17[56 * v16];
    if ( !v20 )
      goto LABEL_24;
    *v20 = (__int64)(v20 + 2);
    v21 = *(_BYTE **)a3;
    v22 = *((_QWORD *)a3 + 1);
    if ( v22 + *(_QWORD *)a3 && !v21 )
      sub_426248((__int64)"basic_string::_M_construct null not valid");
    v42 = *((_QWORD *)a3 + 1);
    if ( v22 > 0xF )
    {
      n = v22;
      src = v21;
      v34 = sub_22409D0(&v17[56 * v16], &v42, 0);
      v21 = src;
      v22 = n;
      *v20 = v34;
      v23 = (_BYTE *)v34;
      v20[2] = v42;
    }
    else
    {
      v23 = (_BYTE *)*v20;
      if ( v22 == 1 )
      {
        *v23 = *v21;
        v22 = v42;
        v23 = (_BYTE *)*v20;
LABEL_18:
        v20[1] = v22;
        v23[v22] = 0;
        v24 = (char *)(*((_QWORD *)a3 + 5) - *((_QWORD *)a3 + 4));
        v20[4] = 0;
        v20[5] = 0;
        v20[6] = 0;
        if ( v24 )
        {
          if ( (unsigned __int64)v24 > 0x7FFFFFFFFFFFFFF8LL )
            sub_4261EA(v23, v17, v24, v19);
          src = v24;
          v25 = sub_22077B0(v24);
          v24 = (char *)src;
          v26 = v25;
        }
        else
        {
          v26 = 0;
        }
        v20[4] = v26;
        v20[5] = v26;
        v20[6] = (__int64)&v24[v26];
        v27 = (_BYTE *)*((_QWORD *)a3 + 5);
        v28 = (_BYTE *)*((_QWORD *)a3 + 4);
        v29 = v27 - v28;
        if ( v27 != v28 )
          v26 = (__int64)memmove((void *)v26, v28, v29);
        v17 = v43;
        v18 = v44;
        v20[5] = v29 + v26;
LABEL_24:
        LODWORD(v44) = v18 + 1;
        v30 = sub_B4BA60(v5, (__int64)v17, (unsigned int)(v18 + 1), v41, v40);
        v31 = v43;
        v5 = (unsigned __int8 *)v30;
        v32 = &v43[56 * (unsigned int)v44];
        if ( v43 != v32 )
        {
          do
          {
            v33 = *((_QWORD *)v32 - 3);
            v32 -= 56;
            if ( v33 )
            {
              v17 = (char *)(*((_QWORD *)v32 + 6) - v33);
              j_j___libc_free_0(v33, v17);
            }
            if ( *(char **)v32 != v32 + 16 )
            {
              v17 = (char *)(*((_QWORD *)v32 + 2) + 1LL);
              j_j___libc_free_0(*(_QWORD *)v32, v17);
            }
          }
          while ( v31 != v32 );
          v32 = v43;
        }
        if ( v32 != v45 )
          _libc_free(v32, v17);
        return v5;
      }
      if ( !v22 )
        goto LABEL_18;
    }
    v17 = v21;
    memcpy(v23, v21, v22);
    v22 = v42;
    v23 = (_BYTE *)*v20;
    goto LABEL_18;
  }
  v12 = 0;
  v13 = 16LL * (unsigned int)v11;
  while ( 1 )
  {
    v14 = 0;
    if ( *(char *)(a1 + 7) < 0 )
      v14 = sub_BD2BC0(a1);
    if ( a2 == *(_DWORD *)(*(_QWORD *)(v14 + v12) + 8LL) )
      return v5;
    v12 += 16;
    if ( v12 == v13 )
      goto LABEL_11;
  }
}
