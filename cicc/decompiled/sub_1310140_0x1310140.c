// Function: sub_1310140
// Address: 0x1310140
//
__int16 __fastcall sub_1310140(__int64 a1, __int64 *a2, __int64 *a3, unsigned int a4, int a5)
{
  __int64 v8; // rsi
  int v9; // edx
  int v10; // ecx
  unsigned __int16 *v11; // r9
  int v12; // eax
  int v13; // eax
  int v14; // r14d
  __int64 v15; // rax
  char v16; // bl
  unsigned int v17; // r13d
  __int64 v18; // rax
  void *v19; // rsp
  void *v20; // rsp
  unsigned __int64 **v21; // r12
  __int64 v22; // rax
  unsigned __int64 *v23; // rax
  int v24; // r8d
  pthread_mutex_t *v25; // rdi
  unsigned __int64 v26; // rax
  __int64 v27; // r10
  __int64 v28; // r14
  int v29; // eax
  int v30; // r8d
  char v31; // al
  __int64 *v32; // rbx
  __int64 v33; // rdi
  __int64 v34; // rbx
  unsigned __int64 **v35; // r15
  unsigned int v36; // r14d
  int v37; // r12d
  __int64 v38; // rdx
  unsigned __int64 *v39; // r10
  __int64 v40; // rcx
  __int64 v41; // r11
  unsigned __int64 *v42; // rsi
  int v43; // r9d
  __int64 v44; // r14
  int v45; // r13d
  bool v46; // sf
  __int64 v47; // rcx
  unsigned __int64 v48; // rdx
  __int64 v49; // rcx
  unsigned __int64 v50; // rdx
  __int64 v51; // rdx
  char v52; // al
  __int64 v53; // rdx
  __int64 v54; // r14
  unsigned __int64 **v55; // r15
  int v56; // r12d
  unsigned __int64 *v57; // rsi
  __int64 v58; // rax
  pthread_mutex_t *v59; // rdi
  int v60; // eax
  __int64 *v61; // rdi
  __int64 v62; // r14
  int v63; // eax
  pthread_mutex_t *v64; // rax
  pthread_mutex_t *v65; // rax
  _QWORD **v66; // r12
  __int64 v67; // rbx
  _QWORD *v68; // rdx
  __int64 v69; // rax
  pthread_mutex_t *v70; // r12
  __int64 v71; // rbx
  _BYTE *v72; // rax
  __int64 *v73; // rsi
  __int16 v74; // dx
  __int64 *v75; // rsi
  _QWORD *v77[2]; // [rsp+0h] [rbp-E0h] BYREF
  unsigned int v78; // [rsp+10h] [rbp-D0h]
  int v79; // [rsp+14h] [rbp-CCh]
  volatile signed __int64 *v80; // [rsp+18h] [rbp-C8h]
  volatile signed __int64 *v81; // [rsp+20h] [rbp-C0h]
  __int16 *v82; // [rsp+28h] [rbp-B8h]
  _QWORD **v83; // [rsp+30h] [rbp-B0h]
  unsigned __int64 *v84; // [rsp+38h] [rbp-A8h]
  __int64 v85; // [rsp+40h] [rbp-A0h]
  unsigned int *v86; // [rsp+48h] [rbp-98h]
  __int64 *v87; // [rsp+50h] [rbp-90h]
  pthread_mutex_t *v88; // [rsp+58h] [rbp-88h]
  int v89; // [rsp+60h] [rbp-80h]
  char v90; // [rsp+67h] [rbp-79h]
  __int64 v91; // [rsp+68h] [rbp-78h]
  __int64 v92; // [rsp+70h] [rbp-70h]
  int v93; // [rsp+78h] [rbp-68h]
  unsigned int v94; // [rsp+7Ch] [rbp-64h]
  __int64 v95; // [rsp+80h] [rbp-60h]
  pthread_mutex_t *v96; // [rsp+88h] [rbp-58h]
  __int64 v97; // [rsp+90h] [rbp-50h]
  __int64 v98; // [rsp+98h] [rbp-48h]
  __int16 v99; // [rsp+A0h] [rbp-40h] BYREF
  __int64 v100; // [rsp+A8h] [rbp-38h]

  v8 = a4;
  v87 = a3;
  v78 = a4;
  v9 = *((unsigned __int16 *)a3 + 10);
  v10 = *((unsigned __int16 *)a3 + 9);
  v79 = a5;
  v11 = (unsigned __int16 *)(unk_5060A20 + 2 * v8);
  LOWORD(v98) = v10;
  v12 = *v11;
  v82 = (__int16 *)v11;
  v13 = v10 - v9 + 8 * v12;
  LOWORD(v13) = (unsigned __int16)v13 >> 3;
  if ( !(_WORD)v13 )
    return v13;
  v14 = v13;
  v15 = *a3;
  v16 = a5;
  v85 = v8;
  v99 = v14;
  v17 = (unsigned __int16)v14;
  v98 = (unsigned __int16)v14;
  v100 = (unsigned __int16)(v9 - v15) + v15 - 8LL * *v11;
  sub_130DB10(v100, (unsigned __int16)v14, qword_505FA40[v8]);
  v18 = *a2;
  v77[1] = v77;
  v88 = *(pthread_mutex_t **)(v18 + 40);
  v19 = alloca((8 * (v14 + 1) + 15) & 0x3FFF0);
  sub_130FEB0((_QWORD *)a1, (__int64)&v99, (unsigned __int16)v14, v77);
  v20 = alloca((8 * (v14 + 1) + 15) & 0x3FFF0);
  v83 = v77;
  v21 = v77;
  v89 = 0;
  v22 = 2 * (3 * v85 - 108);
  v90 = 0;
  LOBYTE(v98) = v16;
  v80 = (volatile signed __int64 *)&v88[25].__size[v22 * 8 + 8];
  v81 = (volatile signed __int64 *)&(&v88[24].__list.__next)[v22];
  v86 = (unsigned int *)((char *)&unk_5260CA0 + 4 * v85);
  v91 = 40 * v85;
  while ( 2 )
  {
    v23 = *v21;
    v24 = *(_DWORD *)*v21 & 0xFFF;
    v25 = (pthread_mutex_t *)qword_50579C0[v24];
    v96 = v25;
    if ( (_BYTE)v98 )
    {
      v26 = *v23;
      LODWORD(v97) = v24;
      v27 = dword_5060A40[v85];
      v93 = (v26 >> 38) & 0x3F;
      v28 = (__int64)v25 + 224 * ((v26 >> 38) & 0x3F) + v27;
      v29 = pthread_mutex_trylock((pthread_mutex_t *)(v28 + 64));
      v30 = v97;
      if ( v29 )
      {
        sub_130AD90(v28);
        *(_BYTE *)(v28 + 104) = 1;
        v30 = v97;
      }
      ++*(_QWORD *)(v28 + 56);
      if ( a1 != *(_QWORD *)(v28 + 48) )
      {
        ++*(_QWORD *)(v28 + 40);
        *(_QWORD *)(v28 + 48) = a1;
      }
      v31 = (v90 ^ 1) & (v88 == v96);
      if ( v31 )
      {
        v32 = v87;
        ++*(_QWORD *)(v28 + 152);
        v90 = v31;
        *(_QWORD *)(v28 + 128) += v32[1];
        v32[1] = 0;
      }
      v94 = *v86;
    }
    else
    {
      if ( v96[1973].__owner >= unk_5057900 )
      {
        LODWORD(v97) = v24;
        v62 = (__int64)(&v96[263].__align + 2);
        v63 = pthread_mutex_trylock(v96 + 265);
        v24 = v97;
        if ( v63 )
        {
          sub_130AD90(v62);
          v96[266].__size[0] = 1;
          v24 = v97;
        }
        v64 = v96;
        ++v96[264].__list.__next;
        if ( (struct __pthread_internal_list *)a1 != v64[264].__list.__prev )
        {
          ++*(&v64[264].__align + 2);
          v64[264].__list.__prev = (struct __pthread_internal_list *)a1;
        }
      }
      v52 = (v90 ^ 1) & (v88 == v96);
      if ( v52 )
      {
        v61 = v87;
        _InterlockedAdd64(v81, v87[1]);
        _InterlockedAdd64(v80, 1u);
        v61[1] = 0;
        v90 = v52;
      }
      v53 = a1;
      v54 = 0;
      v55 = v21;
      v56 = v24;
      do
      {
        while ( 1 )
        {
          v57 = v55[v54];
          if ( v56 == (*(_DWORD *)v57 & 0xFFF) )
            break;
          if ( v17 <= (unsigned int)++v54 )
            goto LABEL_36;
        }
        ++v54;
        v97 = v53;
        sub_130A0D0(v53, v57);
        v53 = v97;
      }
      while ( v17 > (unsigned int)v54 );
LABEL_36:
      v30 = v56;
      v21 = v55;
      a1 = v53;
      if ( v96[1973].__owner >= unk_5057900 )
      {
        v65 = v96;
        LODWORD(v97) = v30;
        v28 = 0;
        v96[266].__size[0] = 0;
        pthread_mutex_unlock(v65 + 265);
        v93 = 0;
        v30 = v97;
      }
      else
      {
        v93 = 0;
        v28 = 0;
      }
      v94 = 0;
    }
    v92 = v28;
    v33 = a1;
    v34 = 0;
    v35 = v21;
    v95 = 0;
    v36 = 0;
    v37 = v30;
    do
    {
      while ( 1 )
      {
        v39 = v35[v34];
        v40 = *(_QWORD *)(v100 + 8 * v34);
        v41 = *v39 & 0xFFF;
        if ( !(_BYTE)v98 )
          break;
        if ( v37 == (_DWORD)v41 && ((*v39 >> 38) & 0x3F) == v93 )
        {
          v47 = v40 - v39[1];
          ++v95;
          v48 = v94 * v47;
          v49 = v91 + 86380000;
          v39[(v48 >> 38) + 8] ^= 1LL << SBYTE4(v48);
          v50 = *v39 + 0x10000000;
          *v39 = v50;
          v51 = (v50 >> 28) & 0x3FF;
          if ( *(_DWORD *)(v49 + 16) == (_DWORD)v51 )
          {
            v97 = v33;
            v84 = v39;
            sub_1315970(v33, v96, v39, v92);
            v60 = v89 + 1;
            v83[v89] = v84;
            v33 = v97;
            v89 = v60;
          }
          else if ( (_DWORD)v51 == 1 && v39 != *(unsigned __int64 **)(v92 + 192) )
          {
            v97 = v33;
            sub_1315A80(v33, v96, v39);
            v33 = v97;
          }
          goto LABEL_14;
        }
LABEL_13:
        v38 = v36++;
        *(_QWORD *)(v100 + 8 * v38) = v40;
        v35[v38] = v39;
LABEL_14:
        if ( v17 <= (unsigned int)++v34 )
          goto LABEL_18;
      }
      if ( v37 != (_DWORD)v41 )
        goto LABEL_13;
      v42 = v35[v34++];
      v97 = v33;
      sub_130A0F0(v33, v42);
      v33 = v97;
    }
    while ( v17 > (unsigned int)v34 );
LABEL_18:
    v43 = v36;
    v21 = v35;
    v44 = v92;
    a1 = v33;
    if ( (_BYTE)v98 )
    {
      v58 = v95;
      LODWORD(v97) = v43;
      v59 = (pthread_mutex_t *)(v92 + 64);
      *(_QWORD *)(v92 + 120) += v95;
      *(_QWORD *)(v44 + 136) -= v58;
      *(_BYTE *)(v44 + 104) = 0;
      pthread_mutex_unlock(v59);
      v43 = v97;
    }
    v45 = v17 - v43;
    if ( a1 )
    {
      v46 = *(_DWORD *)(a1 + 152) - v45 < 0;
      *(_DWORD *)(a1 + 152) -= v45;
      if ( v46 )
      {
        if ( (unsigned __int8)sub_130FCA0((_DWORD *)(a1 + 152), (unsigned __int64 *)(a1 + 112)) )
        {
          LODWORD(v97) = v43;
          sub_1315160(a1, v96, 0, 0);
          v43 = v97;
        }
      }
    }
    if ( v43 )
    {
      v17 = v43;
      continue;
    }
    break;
  }
  if ( v89 )
  {
    v66 = v83;
    v67 = (__int64)&v83[(unsigned int)(v89 - 1) + 1];
    do
    {
      v68 = *v66++;
      sub_13152A0(a1, qword_50579C0[*v68 & 0xFFFLL]);
    }
    while ( (_QWORD **)v67 != v66 );
  }
  if ( !v90 )
  {
    if ( (_BYTE)v79 )
    {
      v69 = sub_1315920(a1, v88, v78, 0);
      v70 = (pthread_mutex_t *)(v69 + 64);
      v71 = v69;
      if ( pthread_mutex_trylock((pthread_mutex_t *)(v69 + 64)) )
      {
        sub_130AD90(v71);
        v72 = (_BYTE *)(v71 + 104);
        *(_BYTE *)(v71 + 104) = 1;
      }
      else
      {
        v72 = (_BYTE *)(v71 + 104);
      }
      ++*(_QWORD *)(v71 + 56);
      if ( a1 != *(_QWORD *)(v71 + 48) )
      {
        ++*(_QWORD *)(v71 + 40);
        *(_QWORD *)(v71 + 48) = a1;
      }
      v73 = v87;
      ++*(_QWORD *)(v71 + 152);
      *(_QWORD *)(v71 + 128) += v73[1];
      v73[1] = 0;
      *v72 = 0;
      pthread_mutex_unlock(v70);
    }
    else
    {
      v75 = v87;
      _InterlockedAdd64(v81, v87[1]);
      _InterlockedAdd64(v80, 1u);
      v75[1] = 0;
    }
  }
  v74 = *v82;
  v98 = *v87;
  LOWORD(v97) = *((_WORD *)v87 + 10);
  LOWORD(v13) = v97 - 8 * v74;
  *((_WORD *)v87 + 9) = v13;
  return v13;
}
