// Function: sub_357B0F0
// Address: 0x357b0f0
//
_QWORD *__fastcall sub_357B0F0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v5; // rsi
  unsigned int v6; // ecx
  __int64 *v7; // rdx
  __int64 v8; // r8
  _QWORD *v9; // r12
  int v11; // edx
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  unsigned int v15; // esi
  __int64 v16; // rcx
  int v17; // r11d
  __int64 *v18; // r8
  unsigned int v19; // edx
  __int64 *v20; // rax
  __int64 v21; // r10
  unsigned __int64 v22; // r14
  unsigned __int64 v23; // r14
  _QWORD *v24; // rbx
  unsigned __int64 v25; // rdi
  int v26; // r9d
  int v27; // eax
  int v28; // edx
  int v29; // eax
  int v30; // esi
  __int64 v31; // rdi
  unsigned int v32; // eax
  __int64 v33; // rcx
  int v34; // r10d
  __int64 *v35; // r9
  int v36; // eax
  int v37; // ecx
  __int64 v38; // rdi
  int v39; // r9d
  unsigned int v40; // r15d
  __int64 *v41; // rsi
  __int64 v42; // rax
  unsigned __int64 v43; // [rsp+8h] [rbp-98h] BYREF
  __int64 v44[5]; // [rsp+10h] [rbp-90h] BYREF
  _QWORD v45[4]; // [rsp+38h] [rbp-68h] BYREF
  unsigned __int64 v46; // [rsp+58h] [rbp-48h]
  __int64 v47; // [rsp+60h] [rbp-40h]

  if ( *(_DWORD *)(a2 + 120) > 1u )
  {
    v2 = *(unsigned int *)(a1 + 432);
    v5 = *(_QWORD *)(a1 + 416);
    if ( (_DWORD)v2 )
    {
      v6 = (v2 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v7 = (__int64 *)(v5 + 16LL * v6);
      v8 = *v7;
      if ( *v7 == a2 )
      {
LABEL_4:
        if ( v7 != (__int64 *)(v5 + 16 * v2) )
          return (_QWORD *)v7[1];
      }
      else
      {
        v11 = 1;
        while ( v8 != -4096 )
        {
          v26 = v11 + 1;
          v6 = (v2 - 1) & (v11 + v6);
          v7 = (__int64 *)(v5 + 16LL * v6);
          v8 = *v7;
          if ( *v7 == a2 )
            goto LABEL_4;
          v11 = v26;
        }
      }
    }
    v12 = *(_QWORD *)(a1 + 400);
    v44[3] = a2;
    v13 = *(_QWORD *)(a1 + 392);
    v44[0] = a1;
    v44[2] = v12;
    v44[1] = v13;
    v44[4] = v12;
    v45[1] = v45;
    v45[0] = v45;
    v45[2] = 0;
    v45[3] = v45;
    v14 = sub_22077B0(0xA0u);
    if ( v14 )
    {
      *(_QWORD *)v14 = 0;
      *(_QWORD *)(v14 + 8) = v14 + 32;
      *(_QWORD *)(v14 + 16) = 4;
      *(_DWORD *)(v14 + 24) = 0;
      *(_BYTE *)(v14 + 28) = 1;
      *(_QWORD *)(v14 + 64) = 0;
      *(_QWORD *)(v14 + 72) = v14 + 96;
      *(_QWORD *)(v14 + 80) = 4;
      *(_DWORD *)(v14 + 88) = 0;
      *(_BYTE *)(v14 + 92) = 1;
      *(_QWORD *)(v14 + 128) = 0;
      *(_QWORD *)(v14 + 136) = 0;
      *(_QWORD *)(v14 + 144) = 0;
      *(_DWORD *)(v14 + 152) = 0;
    }
    v46 = v14;
    v47 = v14 + 128;
    sub_3579EE0(&v43, v44);
    v15 = *(_DWORD *)(a1 + 432);
    if ( v15 )
    {
      v16 = *(_QWORD *)(a1 + 416);
      v17 = 1;
      v18 = 0;
      v19 = (v15 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v20 = (__int64 *)(v16 + 16LL * v19);
      v21 = *v20;
      if ( *v20 == a2 )
      {
LABEL_16:
        v22 = v43;
        v9 = (_QWORD *)v20[1];
        if ( v43 )
        {
          sub_C7D6A0(*(_QWORD *)(v43 + 136), 16LL * *(unsigned int *)(v43 + 152), 8);
          if ( !*(_BYTE *)(v22 + 92) )
            _libc_free(*(_QWORD *)(v22 + 72));
          if ( !*(_BYTE *)(v22 + 28) )
            _libc_free(*(_QWORD *)(v22 + 8));
          j_j___libc_free_0(v22);
        }
LABEL_22:
        v23 = v46;
        if ( v46 )
        {
          sub_C7D6A0(*(_QWORD *)(v46 + 136), 16LL * *(unsigned int *)(v46 + 152), 8);
          if ( !*(_BYTE *)(v23 + 92) )
            _libc_free(*(_QWORD *)(v23 + 72));
          if ( !*(_BYTE *)(v23 + 28) )
            _libc_free(*(_QWORD *)(v23 + 8));
          j_j___libc_free_0(v23);
        }
        v24 = (_QWORD *)v45[0];
        while ( v24 != v45 )
        {
          v25 = (unsigned __int64)v24;
          v24 = (_QWORD *)*v24;
          j_j___libc_free_0(v25);
        }
        return v9;
      }
      while ( v21 != -4096 )
      {
        if ( v21 == -8192 && !v18 )
          v18 = v20;
        v19 = (v15 - 1) & (v17 + v19);
        v20 = (__int64 *)(v16 + 16LL * v19);
        v21 = *v20;
        if ( *v20 == a2 )
          goto LABEL_16;
        ++v17;
      }
      if ( !v18 )
        v18 = v20;
      v27 = *(_DWORD *)(a1 + 424);
      ++*(_QWORD *)(a1 + 408);
      v28 = v27 + 1;
      if ( 4 * (v27 + 1) < 3 * v15 )
      {
        if ( v15 - *(_DWORD *)(a1 + 428) - v28 > v15 >> 3 )
        {
LABEL_43:
          *(_DWORD *)(a1 + 424) = v28;
          if ( *v18 != -4096 )
            --*(_DWORD *)(a1 + 428);
          *v18 = a2;
          v9 = (_QWORD *)v43;
          v18[1] = v43;
          goto LABEL_22;
        }
        sub_3579150(a1 + 408, v15);
        v36 = *(_DWORD *)(a1 + 432);
        if ( v36 )
        {
          v37 = v36 - 1;
          v38 = *(_QWORD *)(a1 + 416);
          v39 = 1;
          v40 = (v36 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
          v28 = *(_DWORD *)(a1 + 424) + 1;
          v41 = 0;
          v18 = (__int64 *)(v38 + 16LL * v40);
          v42 = *v18;
          if ( *v18 != a2 )
          {
            while ( v42 != -4096 )
            {
              if ( v42 == -8192 && !v41 )
                v41 = v18;
              v40 = v37 & (v39 + v40);
              v18 = (__int64 *)(v38 + 16LL * v40);
              v42 = *v18;
              if ( *v18 == a2 )
                goto LABEL_43;
              ++v39;
            }
            if ( v41 )
              v18 = v41;
          }
          goto LABEL_43;
        }
LABEL_70:
        ++*(_DWORD *)(a1 + 424);
        BUG();
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 408);
    }
    sub_3579150(a1 + 408, 2 * v15);
    v29 = *(_DWORD *)(a1 + 432);
    if ( v29 )
    {
      v30 = v29 - 1;
      v31 = *(_QWORD *)(a1 + 416);
      v32 = (v29 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v28 = *(_DWORD *)(a1 + 424) + 1;
      v18 = (__int64 *)(v31 + 16LL * v32);
      v33 = *v18;
      if ( *v18 != a2 )
      {
        v34 = 1;
        v35 = 0;
        while ( v33 != -4096 )
        {
          if ( !v35 && v33 == -8192 )
            v35 = v18;
          v32 = v30 & (v34 + v32);
          v18 = (__int64 *)(v31 + 16LL * v32);
          v33 = *v18;
          if ( *v18 == a2 )
            goto LABEL_43;
          ++v34;
        }
        if ( v35 )
          v18 = v35;
      }
      goto LABEL_43;
    }
    goto LABEL_70;
  }
  v9 = qword_503EFE0;
  if ( !byte_503EFC8[0] && (unsigned int)sub_2207590((__int64)byte_503EFC8) )
  {
    BYTE4(qword_503EFE0[3]) = 1;
    qword_503EFE0[1] = &qword_503EFE0[4];
    qword_503EFE0[0] = 0;
    qword_503EFE0[2] = 4;
    LODWORD(qword_503EFE0[3]) = 0;
    qword_503EFE0[8] = 0;
    qword_503EFE0[9] = &qword_503EFE0[12];
    qword_503EFE0[10] = 4;
    LODWORD(qword_503EFE0[11]) = 0;
    BYTE4(qword_503EFE0[11]) = 1;
    qword_503EFE0[16] = 0;
    qword_503EFE0[17] = 0;
    qword_503EFE0[18] = 0;
    LODWORD(qword_503EFE0[19]) = 0;
    __cxa_atexit((void (*)(void *))sub_3573DF0, qword_503EFE0, &qword_4A427C0);
    sub_2207640((__int64)byte_503EFC8);
  }
  return v9;
}
