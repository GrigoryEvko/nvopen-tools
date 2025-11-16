// Function: sub_2585760
// Address: 0x2585760
//
__int64 __fastcall sub_2585760(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  char v4; // r8
  __int64 result; // rax
  unsigned __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // r9
  __int64 v10; // rax
  void *v11; // r9
  _BYTE *v12; // r8
  __int64 v13; // r14
  unsigned __int64 v14; // rax
  unsigned int v15; // r15d
  _BYTE *v16; // rdi
  __int64 v17; // r15
  char v18; // al
  __int64 v19; // rax
  __int64 v20; // rbx
  unsigned __int64 v21; // rcx
  __int64 v22; // r13
  void **v23; // rdx
  void *v24; // r15
  __int64 v25; // r12
  void **v26; // rdx
  int v27; // eax
  unsigned __int64 v28; // rbx
  void *v29; // r14
  void **v30; // rax
  void **i; // rdx
  size_t v32; // rdx
  int v33; // eax
  __int64 v34; // [rsp+0h] [rbp-1E8h]
  __int64 v35; // [rsp+10h] [rbp-1D8h]
  unsigned __int64 v36; // [rsp+20h] [rbp-1C8h]
  unsigned int v37; // [rsp+30h] [rbp-1B8h]
  int v38; // [rsp+30h] [rbp-1B8h]
  unsigned int v39; // [rsp+30h] [rbp-1B8h]
  char v40; // [rsp+47h] [rbp-1A1h] BYREF
  __m128i v41; // [rsp+48h] [rbp-1A0h] BYREF
  __int64 (__fastcall *v42)(unsigned __int64 *, unsigned __int64 *, int); // [rsp+58h] [rbp-190h]
  __int64 *(__fastcall *v43)(__int64 **, __int64, __int64, __int64 *); // [rsp+60h] [rbp-188h]
  _QWORD v44[4]; // [rsp+68h] [rbp-180h] BYREF
  _BYTE *v45; // [rsp+88h] [rbp-160h] BYREF
  __int64 v46; // [rsp+90h] [rbp-158h]
  _BYTE v47[128]; // [rsp+98h] [rbp-150h] BYREF
  void **p_src; // [rsp+118h] [rbp-D0h] BYREF
  __int64 v49; // [rsp+120h] [rbp-C8h]
  void *src; // [rsp+128h] [rbp-C0h] BYREF
  __int64 v51; // [rsp+130h] [rbp-B8h]
  _BYTE v52[176]; // [rsp+138h] [rbp-B0h] BYREF

  v2 = a2;
  v45 = v47;
  v46 = 0x1000000000LL;
  v40 = 0;
  LODWORD(v44[0]) = 56;
  p_src = (void **)&v45;
  v4 = sub_2526370(
         a2,
         (__int64 (__fastcall *)(__int64, unsigned __int64, __int64))sub_25355B0,
         (__int64)&p_src,
         a1,
         (int *)v44,
         1,
         &v40,
         0,
         0);
  result = 1;
  if ( !v4 )
    goto LABEL_2;
  v36 = sub_250C680((__int64 *)(a1 + 72));
  v6 = sub_250D2C0(v36, 0);
  v8 = sub_2584D90(a2, v6, v7, a1, 2, 0, 1);
  p_src = (void **)a1;
  v35 = v8;
  src = v52;
  v49 = v36;
  v51 = 0x1000000000LL;
  if ( (_DWORD)v46 )
    sub_2538470((__int64)&src, (__int64)&v45, v34, (unsigned int)v46, (__int64)v52, v9);
  v42 = 0;
  v10 = sub_22077B0(0xA0u);
  v12 = v52;
  v13 = v10;
  if ( v10 )
  {
    v14 = (unsigned __int64)p_src;
    v15 = v51;
    v11 = (void *)(v13 + 32);
    *(_QWORD *)(v13 + 16) = v13 + 32;
    v16 = src;
    *(_QWORD *)v13 = v14;
    *(_QWORD *)(v13 + 8) = v49;
    *(_QWORD *)(v13 + 24) = 0x1000000000LL;
    if ( v15 )
    {
      if ( v16 != v52 )
      {
        v33 = HIDWORD(v51);
        *(_QWORD *)(v13 + 16) = v16;
        *(_DWORD *)(v13 + 24) = v15;
        *(_DWORD *)(v13 + 28) = v33;
        v43 = sub_2555B50;
        v41.m128i_i64[0] = v13;
        v42 = sub_25395A0;
        goto LABEL_11;
      }
      v32 = 8LL * v15;
      if ( v15 <= 0x10
        || (sub_C8D5F0(v13 + 16, (const void *)(v13 + 32), v15, 8u, (__int64)v52, (__int64)v11),
            v11 = *(void **)(v13 + 16),
            v16 = src,
            v12 = v52,
            (v32 = 8LL * (unsigned int)v51) != 0) )
      {
        memcpy(v11, v16, v32);
        v16 = src;
        v12 = v52;
      }
      *(_DWORD *)(v13 + 24) = v15;
    }
  }
  else
  {
    v16 = src;
  }
  v41.m128i_i64[0] = v13;
  v43 = sub_2555B50;
  v42 = sub_25395A0;
  if ( v16 != v52 )
    _libc_free((unsigned __int64)v16);
LABEL_11:
  v17 = *(_QWORD *)(a1 + 104);
  v44[1] = a1;
  v44[0] = v35;
  v44[3] = sub_2556790;
  v44[2] = sub_2535AB0;
  p_src = &src;
  v49 = 0x1000000000LL;
  v18 = *(_BYTE *)(v17 + 8);
  if ( v18 == 15 )
  {
    v19 = *(unsigned int *)(v17 + 12);
    if ( (_DWORD)v19 )
    {
      v20 = 8;
      v21 = 0;
      v22 = v17;
      v23 = &src;
      v24 = **(void ***)(v17 + 16);
      v25 = 8 * v19;
      while ( 1 )
      {
        v23[v21] = v24;
        v21 = (unsigned int)(v49 + 1);
        LODWORD(v49) = v49 + 1;
        if ( v25 == v20 )
          break;
        v24 = *(void **)(*(_QWORD *)(v22 + 16) + v20);
        if ( v21 + 1 > HIDWORD(v49) )
        {
          sub_C8D5F0((__int64)&p_src, &src, v21 + 1, 8u, (__int64)v12, (__int64)v11);
          v21 = (unsigned int)v49;
        }
        v23 = p_src;
        v20 += 8;
      }
      v26 = p_src;
      v2 = a2;
    }
    else
    {
      v26 = &src;
      v21 = 0;
    }
    goto LABEL_19;
  }
  if ( v18 != 16 )
  {
    src = (void *)v17;
    v26 = &src;
    v21 = 1;
    LODWORD(v49) = 1;
    goto LABEL_19;
  }
  v28 = *(_QWORD *)(v17 + 32);
  v29 = *(void **)(v17 + 24);
  if ( v28 > 0x10 )
  {
    sub_C8D5F0((__int64)&p_src, &src, *(_QWORD *)(v17 + 32), 8u, (__int64)v12, (__int64)v11);
    v30 = &p_src[(unsigned int)v49];
    goto LABEL_25;
  }
  v30 = &src;
  v26 = &src;
  if ( v28 )
  {
LABEL_25:
    for ( i = &v30[v28]; i != v30; ++v30 )
      *v30 = v29;
    v26 = p_src;
    LODWORD(v28) = v49 + v28;
  }
  LODWORD(v49) = v28;
  v21 = (unsigned int)v28;
LABEL_19:
  v27 = (unsigned __int8)sub_25139F0(v2, v36, v26, v21, &v41, (__int64)v44) ^ 1;
  if ( p_src != &src )
  {
    v38 = v27;
    _libc_free((unsigned __int64)p_src);
    v27 = v38;
  }
  v39 = v27;
  sub_A17130((__int64)v44);
  sub_A17130((__int64)&v41);
  result = v39;
LABEL_2:
  if ( v45 != v47 )
  {
    v37 = result;
    _libc_free((unsigned __int64)v45);
    return v37;
  }
  return result;
}
