// Function: sub_BC4450
// Address: 0xbc4450
//
_QWORD *__fastcall sub_BC4450(__int64 *a1)
{
  __int64 v2; // rbx
  __int64 v3; // r13
  unsigned int v4; // eax
  __int64 v5; // rsi
  __int64 v6; // r8
  __int64 v7; // rdi
  _QWORD *v8; // r9
  int v9; // r15d
  __int64 *v10; // rcx
  _QWORD *v11; // rax
  __int64 v12; // rdx
  _QWORD *v13; // r15
  int v15; // eax
  __int64 v16; // r15
  size_t v17; // rdx
  size_t v18; // r14
  __int64 v19; // rax
  size_t v20; // r12
  const void *v21; // r13
  unsigned int v22; // eax
  unsigned int v23; // r15d
  __int64 *v24; // rcx
  __int64 v25; // rax
  unsigned int v26; // edx
  _QWORD *v27; // rcx
  void *v28; // r8
  _QWORD *v29; // rax
  _QWORD *v30; // rax
  _QWORD *v31; // r12
  __int64 v32; // rax
  __int64 *v33; // rcx
  __int64 v34; // r14
  __int64 *v35; // rdx
  __int64 *v36; // rdx
  size_t v37; // r15
  _QWORD *v38; // rax
  int v39; // eax
  unsigned int v40; // eax
  __int64 *v41; // rdi
  int v42; // eax
  int v43; // eax
  int v44; // edi
  unsigned int v45; // r14d
  __int64 v46; // rax
  _QWORD *v47; // rdi
  _QWORD *v48; // [rsp+0h] [rbp-120h]
  void *src; // [rsp+8h] [rbp-118h]
  __int64 *srca; // [rsp+8h] [rbp-118h]
  _QWORD *v51; // [rsp+10h] [rbp-110h]
  pthread_mutex_t *mutex; // [rsp+18h] [rbp-108h]
  _BYTE *v53; // [rsp+20h] [rbp-100h] BYREF
  size_t n; // [rsp+28h] [rbp-F8h]
  _QWORD *v55; // [rsp+30h] [rbp-F0h] BYREF
  void *v56; // [rsp+38h] [rbp-E8h]
  _QWORD v57[2]; // [rsp+40h] [rbp-E0h] BYREF
  _QWORD v58[2]; // [rsp+50h] [rbp-D0h] BYREF
  __int64 v59; // [rsp+60h] [rbp-C0h]
  __int64 v60; // [rsp+68h] [rbp-B8h]
  __int64 v61; // [rsp+70h] [rbp-B0h]
  __int64 v62; // [rsp+78h] [rbp-A8h]
  _QWORD *v63; // [rsp+80h] [rbp-A0h]
  _QWORD v64[4]; // [rsp+90h] [rbp-90h] BYREF
  char v65; // [rsp+B0h] [rbp-70h]
  _QWORD v66[2]; // [rsp+B8h] [rbp-68h] BYREF
  _QWORD v67[2]; // [rsp+C8h] [rbp-58h] BYREF
  _QWORD v68[9]; // [rsp+D8h] [rbp-48h] BYREF

  v2 = qword_4F824E8;
  if ( !qword_4F824E8 )
  {
    if ( !byte_4F826E9[0] )
      return 0;
    if ( !qword_4F824F0 )
      sub_C7D570(&qword_4F824F0, sub_BC35B0, sub_BC36F0);
    v2 = qword_4F824F0;
    qword_4F824E8 = qword_4F824F0;
    if ( !qword_4F824F0 )
      return 0;
  }
  v3 = (*(__int64 (__fastcall **)(__int64 *))(*a1 + 120))(a1);
  if ( v3 )
    return 0;
  if ( !qword_4F824E8 && byte_4F826E9[0] )
  {
    if ( !qword_4F824F0 )
      sub_C7D570(&qword_4F824F0, sub_BC35B0, sub_BC36F0);
    qword_4F824E8 = qword_4F824F0;
  }
  if ( !qword_4F82510 )
    sub_C7D570(&qword_4F82510, sub_BC3580, sub_BC3540);
  mutex = qword_4F82510;
  if ( &_pthread_key_create )
  {
    v4 = pthread_mutex_lock(qword_4F82510);
    if ( v4 )
      sub_4264C5(v4);
  }
  v5 = *(unsigned int *)(v2 + 48);
  v6 = v2 + 24;
  if ( !(_DWORD)v5 )
  {
    ++*(_QWORD *)(v2 + 24);
    goto LABEL_67;
  }
  v7 = *(_QWORD *)(v2 + 32);
  v8 = 0;
  v9 = 1;
  v10 = (__int64 *)(((_DWORD)v5 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4)));
  v11 = (_QWORD *)(v7 + 16LL * (_QWORD)v10);
  v12 = *v11;
  if ( a1 != (__int64 *)*v11 )
  {
    while ( v12 != -4096 )
    {
      if ( v12 == -8192 && !v8 )
        v8 = v11;
      v10 = (__int64 *)(((_DWORD)v5 - 1) & (unsigned int)(v9 + (_DWORD)v10));
      v11 = (_QWORD *)(v7 + 16LL * (unsigned int)v10);
      v12 = *v11;
      if ( a1 == (__int64 *)*v11 )
        goto LABEL_10;
      ++v9;
    }
    if ( !v8 )
      v8 = v11;
    v15 = *(_DWORD *)(v2 + 40);
    ++*(_QWORD *)(v2 + 24);
    v12 = (unsigned int)(v15 + 1);
    if ( 4 * (int)v12 < (unsigned int)(3 * v5) )
    {
      v10 = (__int64 *)((unsigned int)v5 >> 3);
      if ( (int)v5 - *(_DWORD *)(v2 + 44) - (int)v12 > (unsigned int)v10 )
      {
LABEL_33:
        *(_DWORD *)(v2 + 40) = v12;
        if ( *v8 != -4096 )
          --*(_DWORD *)(v2 + 44);
        *v8 = a1;
        v8[1] = 0;
        v51 = v8 + 1;
LABEL_36:
        v16 = (*(__int64 (__fastcall **)(__int64 *, __int64, __int64, __int64 *, __int64, _QWORD *))(*a1 + 16))(
                a1,
                v5,
                v12,
                v10,
                v6,
                v8);
        v18 = v17;
        v19 = sub_BB9590(a1[2], v5);
        if ( !v19 || (v20 = *(_QWORD *)(v19 + 24), v21 = *(const void **)(v19 + 16), !v20) )
        {
          v21 = (const void *)v16;
          v20 = v18;
        }
        v53 = (_BYTE *)v16;
        n = v18;
        v22 = sub_C92610(v21, v20);
        v23 = sub_C92740(v2, v21, v20, v22);
        v24 = (__int64 *)(*(_QWORD *)v2 + 8LL * v23);
        v25 = *v24;
        if ( *v24 )
        {
          if ( v25 != -8 )
            goto LABEL_40;
          --*(_DWORD *)(v2 + 16);
        }
        srca = v24;
        v32 = sub_C7D670(v20 + 17, 8);
        v33 = srca;
        v34 = v32;
        if ( v20 )
        {
          memcpy((void *)(v32 + 16), v21, v20);
          v33 = srca;
        }
        *(_BYTE *)(v34 + v20 + 16) = 0;
        *(_QWORD *)v34 = v20;
        *(_DWORD *)(v34 + 8) = 0;
        *v33 = v34;
        ++*(_DWORD *)(v2 + 12);
        v35 = (__int64 *)(*(_QWORD *)v2 + 8LL * (unsigned int)sub_C929D0(v2, v23));
        v25 = *v35;
        if ( *v35 == -8 || !v25 )
        {
          v36 = v35 + 1;
          do
          {
            do
              v25 = *v36++;
            while ( !v25 );
          }
          while ( v25 == -8 );
        }
LABEL_40:
        v26 = *(_DWORD *)(v25 + 8) + 1;
        *(_DWORD *)(v25 + 8) = v26;
        if ( v26 > 1 )
        {
          v66[1] = v25 + 8;
          v64[2] = v68;
          v65 = 1;
          v67[0] = &unk_49DB108;
          v67[1] = &v53;
          v64[0] = "{0} #{1}";
          v66[0] = &unk_49DB138;
          v68[0] = v67;
          v68[1] = v66;
          v62 = 0x100000000LL;
          v58[0] = &unk_49DD210;
          v63 = &v55;
          v64[1] = 8;
          v64[3] = 2;
          v55 = v57;
          v56 = 0;
          LOBYTE(v57[0]) = 0;
          v58[1] = 0;
          v59 = 0;
          v60 = 0;
          v61 = 0;
          sub_CB5980(v58, 0, 0, 0);
          sub_CB6840(v58, v64);
          if ( v61 != v59 )
            sub_CB5AE0(v58);
          v58[0] = &unk_49DD210;
          sub_CB5840(v58);
          v27 = v55;
          v28 = v56;
          goto LABEL_44;
        }
        v28 = v53;
        if ( !v53 )
        {
          LOBYTE(v57[0]) = 0;
          v55 = v57;
          v27 = v57;
          v56 = 0;
          goto LABEL_44;
        }
        v37 = n;
        v55 = v57;
        v64[0] = n;
        if ( n > 0xF )
        {
          v46 = sub_22409D0(&v55, v64, 0);
          v28 = v53;
          v55 = (_QWORD *)v46;
          v47 = (_QWORD *)v46;
          v57[0] = v64[0];
        }
        else
        {
          if ( n == 1 )
          {
            LOBYTE(v57[0]) = *v53;
            v38 = v57;
LABEL_64:
            v56 = (void *)v37;
            *((_BYTE *)v38 + v37) = 0;
            v27 = v55;
            v28 = v56;
LABEL_44:
            v48 = v27;
            src = v28;
            v29 = (_QWORD *)sub_22077B0(176);
            v13 = v29;
            if ( v29 )
            {
              *v29 = 0;
              v30 = v29 + 12;
              *(v30 - 11) = 0;
              *(v30 - 10) = 0;
              *(v30 - 9) = 0;
              *(v30 - 8) = 0;
              *(v30 - 7) = 0;
              *(v30 - 6) = 0;
              *(v30 - 5) = 0;
              *(v30 - 4) = 0;
              *(v30 - 3) = 0;
              v13[10] = v30;
              v13[14] = v13 + 16;
              v13[11] = 0;
              *((_BYTE *)v13 + 96) = 0;
              v13[15] = 0;
              *((_BYTE *)v13 + 128) = 0;
              *((_WORD *)v13 + 72) = 0;
              v13[19] = 0;
              v13[20] = 0;
              v13[21] = 0;
              sub_C9EA20(v13, v21, v20, v48, src, v2 + 56);
            }
            if ( v55 != v57 )
              j_j___libc_free_0(v55, v57[0] + 1LL);
            v31 = (_QWORD *)*v51;
            *v51 = v13;
            if ( v31 )
            {
              sub_C9F8C0(v31);
              j_j___libc_free_0(v31, 176);
              v13 = (_QWORD *)*v51;
            }
            goto LABEL_11;
          }
          if ( !n )
          {
            v38 = v57;
            goto LABEL_64;
          }
          v47 = v57;
        }
        memcpy(v47, v28, n);
        v37 = v64[0];
        v38 = v55;
        goto LABEL_64;
      }
      sub_BC4240(v2 + 24, v5);
      v42 = *(_DWORD *)(v2 + 48);
      if ( v42 )
      {
        v43 = v42 - 1;
        v5 = *(_QWORD *)(v2 + 32);
        v44 = 1;
        v45 = v43 & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
        v8 = (_QWORD *)(v5 + 16LL * v45);
        v10 = (__int64 *)*v8;
        v12 = (unsigned int)(*(_DWORD *)(v2 + 40) + 1);
        if ( a1 == (__int64 *)*v8 )
          goto LABEL_33;
        while ( v10 != (__int64 *)-4096LL )
        {
          if ( v10 == (__int64 *)-8192LL && !v3 )
            v3 = (__int64)v8;
          v6 = (unsigned int)(v44 + 1);
          v45 = v43 & (v44 + v45);
          v8 = (_QWORD *)(v5 + 16LL * v45);
          v10 = (__int64 *)*v8;
          if ( a1 == (__int64 *)*v8 )
            goto LABEL_33;
          ++v44;
        }
        goto LABEL_71;
      }
      goto LABEL_93;
    }
LABEL_67:
    sub_BC4240(v2 + 24, 2 * v5);
    v39 = *(_DWORD *)(v2 + 48);
    if ( v39 )
    {
      v10 = (__int64 *)(unsigned int)(v39 - 1);
      v5 = *(_QWORD *)(v2 + 32);
      v40 = (unsigned int)v10 & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
      v8 = (_QWORD *)(v5 + 16LL * v40);
      v41 = (__int64 *)*v8;
      v12 = (unsigned int)(*(_DWORD *)(v2 + 40) + 1);
      if ( a1 == (__int64 *)*v8 )
        goto LABEL_33;
      v6 = 1;
      while ( v41 != (__int64 *)-4096LL )
      {
        if ( !v3 && v41 == (__int64 *)-8192LL )
          v3 = (__int64)v8;
        v40 = (unsigned int)v10 & (v6 + v40);
        v8 = (_QWORD *)(v5 + 16LL * v40);
        v41 = (__int64 *)*v8;
        if ( a1 == (__int64 *)*v8 )
          goto LABEL_33;
        v6 = (unsigned int)(v6 + 1);
      }
LABEL_71:
      if ( v3 )
        v8 = (_QWORD *)v3;
      goto LABEL_33;
    }
LABEL_93:
    ++*(_DWORD *)(v2 + 40);
    BUG();
  }
LABEL_10:
  v13 = (_QWORD *)v11[1];
  v5 = (__int64)(v11 + 1);
  v51 = v11 + 1;
  if ( !v13 )
    goto LABEL_36;
LABEL_11:
  if ( &_pthread_key_create )
    pthread_mutex_unlock(mutex);
  return v13;
}
