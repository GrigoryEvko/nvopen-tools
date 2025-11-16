// Function: sub_13108D0
// Address: 0x13108d0
//
unsigned __int16 __fastcall sub_13108D0(__int64 a1, __int64 *a2, __int64 *a3, unsigned int a4, int a5)
{
  __int64 *v5; // r15
  _QWORD *v6; // r14
  __int64 v8; // rdi
  __int16 v9; // ax
  signed __int64 v10; // rbx
  void *v11; // rsp
  void *v12; // rsp
  _QWORD *v13; // rax
  int v14; // r15d
  __int64 v15; // r12
  __int64 v16; // r12
  _QWORD *v17; // rax
  __int64 *v18; // rbx
  _QWORD *v19; // r9
  __int64 v20; // rbx
  unsigned int v21; // r8d
  unsigned int v22; // r12d
  int v23; // r13d
  __int64 v24; // r14
  unsigned int v25; // r10d
  _QWORD **v26; // r15
  __int64 v27; // rsi
  unsigned __int64 *v28; // rdx
  __int64 v29; // rcx
  unsigned __int64 v30; // rsi
  __int64 v31; // rsi
  unsigned __int64 v32; // rcx
  __int64 v33; // rcx
  _BYTE *v34; // rax
  unsigned int v35; // r13d
  _QWORD *v36; // rbx
  unsigned int v37; // r8d
  bool v38; // sf
  int v39; // eax
  _QWORD **v40; // r12
  __int64 v41; // rbx
  _QWORD *v42; // rdx
  __int64 *v43; // r14
  char *v44; // rsi
  __int64 v45; // rbx
  __int16 v46; // ax
  __int64 v47; // rbx
  unsigned __int16 v48; // cx
  unsigned __int16 result; // ax
  __int64 v50; // rax
  pthread_mutex_t *v51; // r12
  __int64 v52; // rbx
  _BYTE *v53; // rax
  __int64 *v54; // rdi
  _QWORD *v55; // [rsp+0h] [rbp-E0h] BYREF
  int v56; // [rsp+Ch] [rbp-D4h]
  _QWORD **v57; // [rsp+10h] [rbp-D0h]
  __int16 v58; // [rsp+1Ah] [rbp-C6h]
  unsigned int v59; // [rsp+1Ch] [rbp-C4h]
  __int64 *v60; // [rsp+20h] [rbp-C0h]
  _QWORD **v61; // [rsp+28h] [rbp-B8h]
  __int64 v62; // [rsp+30h] [rbp-B0h]
  unsigned int *v63; // [rsp+38h] [rbp-A8h]
  __int64 v64; // [rsp+40h] [rbp-A0h]
  unsigned int v65; // [rsp+48h] [rbp-98h]
  int v66; // [rsp+4Ch] [rbp-94h]
  unsigned __int8 v67; // [rsp+53h] [rbp-8Dh]
  unsigned int v68; // [rsp+54h] [rbp-8Ch]
  _QWORD *v69; // [rsp+58h] [rbp-88h]
  unsigned __int64 *v70; // [rsp+60h] [rbp-80h]
  _BYTE *v71; // [rsp+68h] [rbp-78h]
  pthread_mutex_t *mutex; // [rsp+70h] [rbp-70h]
  _QWORD **v73; // [rsp+78h] [rbp-68h]
  __int64 v74; // [rsp+80h] [rbp-60h]
  unsigned int v75; // [rsp+88h] [rbp-58h]
  unsigned int v76; // [rsp+8Ch] [rbp-54h]
  _QWORD *v77; // [rsp+90h] [rbp-50h]
  __int64 v78; // [rsp+98h] [rbp-48h]
  unsigned __int16 v79; // [rsp+A0h] [rbp-40h] BYREF
  __int64 v80; // [rsp+A8h] [rbp-38h]

  v5 = a3;
  v6 = (_QWORD *)a1;
  v56 = a5;
  v60 = a3;
  v59 = a4;
  sub_1310140(a1, a2, a3, a4, 1);
  v8 = *v5;
  v9 = *((_WORD *)v5 + 10);
  v57 = &v55;
  v78 = v8;
  LOWORD(v77) = v9;
  LOWORD(v8) = v9 - v8;
  v58 = (unsigned __int16)v8 >> 3;
  LODWORD(v5) = ((unsigned __int16)v8 >> 3) - a5;
  v75 = (unsigned int)v5;
  v79 = ((unsigned __int16)v8 >> 3) - a5;
  v80 = v78 + (unsigned __int16)v8 - 8LL * v79;
  v64 = *(_QWORD *)(*a2 + 40);
  v10 = 16 * ((8 * (unsigned __int64)(unsigned int)((_DWORD)v5 + 1) + 15) >> 4);
  v11 = alloca(v10);
  sub_130FEB0(v6, (__int64)&v79, (unsigned int)v5, &v55);
  v12 = alloca(v10);
  v61 = &v55;
  if ( !(_DWORD)v5 )
  {
LABEL_33:
    v50 = sub_1315920(v6, v64, v59, 0);
    v51 = (pthread_mutex_t *)(v50 + 64);
    v52 = v50;
    if ( pthread_mutex_trylock((pthread_mutex_t *)(v50 + 64)) )
    {
      sub_130AD90(v52);
      v53 = (_BYTE *)(v52 + 104);
      *(_BYTE *)(v52 + 104) = 1;
    }
    else
    {
      v53 = (_BYTE *)(v52 + 104);
    }
    ++*(_QWORD *)(v52 + 56);
    if ( v6 != *(_QWORD **)(v52 + 48) )
    {
      ++*(_QWORD *)(v52 + 40);
      *(_QWORD *)(v52 + 48) = v6;
    }
    v54 = v60;
    ++*(_QWORD *)(v52 + 152);
    *(_QWORD *)(v52 + 128) += v54[1];
    v54[1] = 0;
    *v53 = 0;
    pthread_mutex_unlock(v51);
    goto LABEL_28;
  }
  v77 = v6;
  v66 = 0;
  v62 = v59;
  v67 = 0;
  v63 = (unsigned int *)((char *)&unk_5260CA0 + 4 * v59);
  v78 = 40LL * v59;
  v73 = &v55;
  while ( 2 )
  {
    v13 = *v73;
    v14 = *(_DWORD *)*v73 & 0xFFF;
    v74 = qword_50579C0[v14];
    v15 = dword_5060A40[v62];
    v76 = (*v13 >> 38) & 0x3F;
    v16 = v15 + v74 + 224LL * v76;
    mutex = (pthread_mutex_t *)(v16 + 64);
    if ( pthread_mutex_trylock((pthread_mutex_t *)(v16 + 64)) )
    {
      sub_130AD90(v16);
      v71 = (_BYTE *)(v16 + 104);
      *(_BYTE *)(v16 + 104) = 1;
    }
    else
    {
      v71 = (_BYTE *)(v16 + 104);
    }
    ++*(_QWORD *)(v16 + 56);
    v17 = v77;
    if ( v77 != *(_QWORD **)(v16 + 48) )
    {
      ++*(_QWORD *)(v16 + 40);
      *(_QWORD *)(v16 + 48) = v17;
    }
    if ( ((v67 ^ 1) & (v64 == v74)) != 0 )
    {
      v18 = v60;
      v67 = (v67 ^ 1) & (v64 == v74);
      ++*(_QWORD *)(v16 + 152);
      *(_QWORD *)(v16 + 128) += v18[1];
      v18[1] = 0;
    }
    v19 = (_QWORD *)v16;
    v20 = 0;
    v21 = 0;
    v22 = v75;
    v23 = v14;
    v24 = 0;
    v25 = *v63;
    v26 = v73;
    do
    {
      while ( 1 )
      {
        v28 = v26[v20];
        v29 = *(_QWORD *)(v80 + 8 * v20);
        if ( v23 == (*(_DWORD *)v28 & 0xFFF) && v76 == ((*v28 >> 38) & 0x3F) )
          break;
        v27 = v21++;
        *(_QWORD *)(v80 + 8 * v27) = v29;
        v26[v27] = v28;
LABEL_11:
        if ( v22 <= (unsigned int)++v20 )
          goto LABEL_18;
      }
      ++v24;
      v30 = v25 * (v29 - v28[1]);
      v28[(v30 >> 38) + 8] ^= 1LL << SBYTE4(v30);
      v31 = v78 + 86380000;
      v32 = *v28 + 0x10000000;
      *v28 = v32;
      v33 = (v32 >> 28) & 0x3FF;
      if ( *(_DWORD *)(v31 + 16) == (_DWORD)v33 )
      {
        v70 = v28;
        v65 = v21;
        v68 = v25;
        v69 = v19;
        sub_1315970(v77, v74, v28, v19);
        v19 = v69;
        v25 = v68;
        v21 = v65;
        v39 = v66 + 1;
        v61[v66] = v70;
        v66 = v39;
        goto LABEL_11;
      }
      if ( (_DWORD)v33 != 1 || v28 == (unsigned __int64 *)v19[24] )
        goto LABEL_11;
      ++v20;
      v68 = v21;
      LODWORD(v69) = v25;
      v70 = v19;
      sub_1315A80(v77, v74, v28);
      v21 = v68;
      v19 = v70;
      v25 = (unsigned int)v69;
    }
    while ( v22 > (unsigned int)v20 );
LABEL_18:
    v34 = v71;
    v19[15] += v24;
    v35 = v21;
    v19[17] -= v24;
    *v34 = 0;
    pthread_mutex_unlock(mutex);
    v36 = v77;
    v37 = v75 - v35;
    if ( v77 )
    {
      v76 = *((_DWORD *)v77 + 38);
      v38 = (int)(v76 - v37) < 0;
      *((_DWORD *)v77 + 38) = v76 - v37;
      if ( v38 )
      {
        if ( (unsigned __int8)sub_130FCA0((_DWORD *)v36 + 38, v36 + 14) )
          sub_1315160(v77, v74, 0, 0);
      }
    }
    if ( v35 )
    {
      v75 = v35;
      continue;
    }
    break;
  }
  v6 = v77;
  if ( v66 )
  {
    v40 = v61;
    v41 = (__int64)&v61[(unsigned int)(v66 - 1) + 1];
    do
    {
      v42 = *v40++;
      sub_13152A0(v6, qword_50579C0[*v42 & 0xFFFLL]);
    }
    while ( (_QWORD **)v41 != v40 );
  }
  if ( !v67 )
    goto LABEL_33;
LABEL_28:
  v43 = v60;
  v44 = (char *)*v60;
  v45 = 8LL * (unsigned __int16)(v58 - v56);
  LOWORD(v78) = *((_WORD *)v60 + 10);
  memmove(
    &v44[v45],
    v44,
    8LL * (((unsigned __int16)(v78 - (_WORD)v44) >> 3) - (unsigned int)(unsigned __int16)(v58 - v56)));
  v46 = *((_WORD *)v43 + 10);
  v47 = *v43 + v45;
  *v43 = v47;
  v48 = v46 - v47;
  result = (unsigned __int16)(v46 - *((_WORD *)v43 + 8)) >> 3;
  if ( (unsigned __int16)(v48 >> 3) < result )
    *((_WORD *)v43 + 8) = v47;
  return result;
}
