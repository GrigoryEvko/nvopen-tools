// Function: sub_2FA48F0
// Address: 0x2fa48f0
//
__int64 __fastcall sub_2FA48F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int64 v5; // r13
  __int64 v7; // r14
  __int64 v8; // rbx
  __int64 v9; // rax
  __int64 v10; // rsi
  int v11; // ecx
  __int64 v12; // rdi
  int v13; // ecx
  unsigned int v14; // edx
  __int64 *v15; // rax
  __int64 v16; // rax
  __int64 v17; // rbx
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r9
  unsigned __int64 v21; // rsi
  _BYTE *v22; // rbx
  unsigned __int64 v23; // rdi
  __int64 v24; // rdx
  _BYTE *v25; // rax
  _BYTE *v26; // r15
  size_t v27; // r9
  __int64 v28; // rbx
  __int64 *v29; // rdi
  int v30; // edx
  __int64 *v31; // r10
  unsigned __int64 v32; // rax
  unsigned __int64 v33; // r12
  __int64 v34; // rdx
  unsigned __int64 *v35; // r14
  unsigned __int64 *v36; // r15
  __int64 *v37; // rbx
  __int64 v38; // r15
  __int64 *v39; // rbx
  __int64 *v40; // r13
  __int64 v41; // rsi
  _BYTE *v42; // r14
  unsigned __int64 v43; // rdi
  _BYTE *v44; // rbx
  unsigned __int64 v45; // r12
  unsigned __int64 v46; // rdi
  int v48; // eax
  int v49; // r9d
  __int64 *v50; // [rsp+8h] [rbp-188h]
  size_t v51; // [rsp+8h] [rbp-188h]
  __int64 v52; // [rsp+20h] [rbp-170h]
  __int64 *v53; // [rsp+20h] [rbp-170h]
  _BYTE *v54; // [rsp+20h] [rbp-170h]
  __int64 *v55; // [rsp+30h] [rbp-160h] BYREF
  __int64 v56; // [rsp+38h] [rbp-158h]
  _BYTE v57[32]; // [rsp+40h] [rbp-150h] BYREF
  _BYTE *v58; // [rsp+60h] [rbp-130h] BYREF
  __int64 v59; // [rsp+68h] [rbp-128h]
  _BYTE v60[112]; // [rsp+70h] [rbp-120h] BYREF
  _BYTE *v61; // [rsp+E0h] [rbp-B0h] BYREF
  __int64 v62; // [rsp+E8h] [rbp-A8h]
  _BYTE v63[160]; // [rsp+F0h] [rbp-A0h] BYREF

  v7 = *(_QWORD *)(a2 + 80);
  v58 = v60;
  v61 = v63;
  v8 = a2 + 72;
  v59 = 0x200000000LL;
  v62 = 0x200000000LL;
  if ( v7 == a2 + 72 )
    goto LABEL_22;
  do
  {
    v9 = *(_QWORD *)(a1 + 32);
    v10 = v7 - 24;
    if ( !v7 )
      v10 = 0;
    v11 = *(_DWORD *)(v9 + 24);
    v12 = *(_QWORD *)(v9 + 8);
    if ( !v11 )
      goto LABEL_8;
    v13 = v11 - 1;
    v14 = v13 & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
    v15 = (__int64 *)(v12 + 16LL * v14);
    a5 = *v15;
    if ( v10 != *v15 )
    {
      v48 = 1;
      while ( a5 != -4096 )
      {
        v49 = v48 + 1;
        v14 = v13 & (v48 + v14);
        v15 = (__int64 *)(v12 + 16LL * v14);
        a5 = *v15;
        if ( v10 == *v15 )
          goto LABEL_6;
        v48 = v49;
      }
LABEL_8:
      sub_2FA1EE0(a1, v10, (__int64)&v61);
      goto LABEL_9;
    }
LABEL_6:
    v16 = v15[1];
    if ( !v16 || *(_QWORD *)(v16 + 16) != *(_QWORD *)(v16 + 8) )
      goto LABEL_8;
LABEL_9:
    v7 = *(_QWORD *)(v7 + 8);
  }
  while ( v8 != v7 );
  v17 = (__int64)v61;
  v5 = (unsigned __int64)&v61[56 * (unsigned int)v62];
  if ( v61 != (_BYTE *)v5 )
  {
    do
    {
      while ( !(unsigned __int8)sub_2F9CF50(a1, v17) )
      {
        v17 += 56;
        if ( v5 == v17 )
          goto LABEL_15;
      }
      v21 = v17;
      v17 += 56;
      sub_2F9A860((__int64)&v58, v21, v18, v19, a5, v20);
    }
    while ( v5 != v17 );
LABEL_15:
    v22 = v61;
    v5 = (unsigned __int64)&v61[56 * (unsigned int)v62];
    if ( v61 != (_BYTE *)v5 )
    {
      do
      {
        v5 -= 56LL;
        v23 = *(_QWORD *)(v5 + 8);
        if ( v23 != v5 + 24 )
          _libc_free(v23);
      }
      while ( v22 != (_BYTE *)v5 );
      v5 = (unsigned __int64)v61;
    }
  }
  if ( (_BYTE *)v5 != v63 )
    _libc_free(v5);
LABEL_22:
  v24 = *(_QWORD *)(a1 + 32);
  v25 = *(_BYTE **)(v24 + 40);
  v26 = *(_BYTE **)(v24 + 32);
  v55 = (__int64 *)v57;
  v56 = 0x400000000LL;
  v27 = v25 - v26;
  v28 = (v25 - v26) >> 3;
  if ( (unsigned __int64)(v25 - v26) > 0x20 )
  {
    v51 = v25 - v26;
    v54 = v25;
    sub_C8D5F0((__int64)&v55, v57, (v25 - v26) >> 3, 8u, a5, v27);
    v29 = v55;
    v30 = v56;
    v25 = v54;
    v27 = v51;
    v31 = &v55[(unsigned int)v56];
  }
  else
  {
    v29 = (__int64 *)v57;
    v30 = 0;
    v31 = (__int64 *)v57;
  }
  if ( v25 != v26 )
  {
    memmove(v31, v26, v27);
    v30 = v56;
    v29 = v55;
  }
  LODWORD(v56) = v28 + v30;
  v32 = (unsigned int)(v28 + v30);
  if ( (_DWORD)v28 + v30 )
  {
    v52 = a1;
    v33 = 0;
    do
    {
      v34 = v29[v33];
      v35 = *(unsigned __int64 **)(v34 + 8);
      v36 = *(unsigned __int64 **)(v34 + 16);
      if ( v35 != v36 )
      {
        do
        {
          v5 = *v35;
          if ( v32 + 1 > HIDWORD(v56) )
          {
            sub_C8D5F0((__int64)&v55, v57, v32 + 1, 8u, a5, v27);
            v32 = (unsigned int)v56;
          }
          ++v35;
          v55[v32] = v5;
          v32 = (unsigned int)(v56 + 1);
          LODWORD(v56) = v56 + 1;
        }
        while ( v36 != v35 );
        v29 = v55;
      }
      ++v33;
    }
    while ( v32 > v33 );
    a1 = v52;
    v53 = &v29[v32];
    if ( v53 != v29 )
    {
      v37 = v29;
      do
      {
        while ( 1 )
        {
          v38 = *v37;
          if ( *(_QWORD *)(*v37 + 16) == *(_QWORD *)(*v37 + 8) )
          {
            v61 = v63;
            v62 = 0x200000000LL;
            if ( *(_QWORD *)(v38 + 32) != *(_QWORD *)(v38 + 40) )
            {
              v50 = v37;
              v39 = *(__int64 **)(v38 + 32);
              v40 = *(__int64 **)(v38 + 40);
              do
              {
                v41 = *v39++;
                sub_2FA1EE0(a1, v41, (__int64)&v61);
              }
              while ( v40 != v39 );
              v37 = v50;
            }
            sub_2F9DAC0(a1, v38, (__int64)&v61, (__int64)&v58);
            v42 = v61;
            v5 = (unsigned __int64)&v61[56 * (unsigned int)v62];
            if ( v61 != (_BYTE *)v5 )
            {
              do
              {
                v5 -= 56LL;
                v43 = *(_QWORD *)(v5 + 8);
                if ( v43 != v5 + 24 )
                  _libc_free(v43);
              }
              while ( v42 != (_BYTE *)v5 );
              v5 = (unsigned __int64)v61;
            }
            if ( (_BYTE *)v5 != v63 )
              break;
          }
          if ( v53 == ++v37 )
            goto LABEL_49;
        }
        ++v37;
        _libc_free(v5);
      }
      while ( v53 != v37 );
LABEL_49:
      v29 = v55;
    }
  }
  if ( v29 != (__int64 *)v57 )
    _libc_free((unsigned __int64)v29);
  sub_2FA2630(a1, (__int64)&v58);
  v44 = v58;
  LOBYTE(v5) = (_DWORD)v59 != 0;
  v45 = (unsigned __int64)&v58[56 * (unsigned int)v59];
  if ( v58 != (_BYTE *)v45 )
  {
    do
    {
      v45 -= 56LL;
      v46 = *(_QWORD *)(v45 + 8);
      if ( v46 != v45 + 24 )
        _libc_free(v46);
    }
    while ( v44 != (_BYTE *)v45 );
    v45 = (unsigned __int64)v58;
  }
  if ( (_BYTE *)v45 != v60 )
    _libc_free(v45);
  return (unsigned int)v5;
}
