// Function: sub_131B220
// Address: 0x131b220
//
_QWORD *__fastcall sub_131B220(
        _BYTE *a1,
        __int64 a2,
        __int64 (__fastcall ***a3)(int, int, int, int, int, int, int),
        int *a4,
        _QWORD *a5,
        __int64 a6,
        __int64 a7)
{
  unsigned __int64 v11; // rax
  unsigned __int64 v12; // rcx
  __int64 v13; // rcx
  __int64 v14; // rsi
  __int64 v15; // rax
  unsigned __int64 v16; // rbx
  __int64 (__fastcall **v17)(int, int, int, int, int, int, int); // r10
  __int64 v18; // rax
  __int64 v19; // rcx
  _BYTE *v20; // r11
  _QWORD *v21; // r8
  __int64 v22; // r9
  unsigned __int64 v24; // rax
  unsigned int v25; // eax
  char v26; // cl
  unsigned int v27; // eax
  int v28; // eax
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v32; // rax
  int v33; // eax
  __int64 v34; // rdx
  __int64 v35; // rcx
  __int64 v36; // r9
  _QWORD *v37; // r8
  _QWORD *v38; // r12
  __int64 v39; // rax
  _BYTE *v40; // rdi
  __int64 v41; // rax
  __int64 v42; // rax
  __int64 v43; // rax
  __int64 v44; // rdi
  __int64 v45; // rax
  __int64 v46; // [rsp-10h] [rbp-70h]
  __int64 v47; // [rsp-10h] [rbp-70h]
  __int64 v48; // [rsp-8h] [rbp-68h]
  __int64 (__fastcall ***v49)(int, int, int, int, int, int, int); // [rsp+8h] [rbp-58h]
  __int64 (__fastcall ***v50)(int, int, int, int, int, int, int); // [rsp+8h] [rbp-58h]
  __int64 (__fastcall ***v51)(int, int, int, int, int, int, int); // [rsp+8h] [rbp-58h]
  pthread_mutex_t *mutexa; // [rsp+10h] [rbp-50h]
  pthread_mutex_t *mutex; // [rsp+10h] [rbp-50h]
  _QWORD *v54; // [rsp+18h] [rbp-48h]
  _QWORD *v55; // [rsp+18h] [rbp-48h]
  _QWORD *v56; // [rsp+18h] [rbp-48h]
  _QWORD *v57; // [rsp+18h] [rbp-48h]
  _QWORD *v58; // [rsp+18h] [rbp-48h]
  unsigned __int64 v59; // [rsp+18h] [rbp-48h]
  __int64 (__fastcall **v60)(_QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD); // [rsp+18h] [rbp-48h]
  _QWORD *v61; // [rsp+18h] [rbp-48h]
  _QWORD *v62; // [rsp+18h] [rbp-48h]
  char v63; // [rsp+2Eh] [rbp-32h] BYREF
  _BYTE v64[49]; // [rsp+2Fh] [rbp-31h] BYREF

  v11 = (-(__int64)((a7 + 15) & 0xFFFFFFFFFFFFFFF0LL) & (((a7 + 15) & 0xFFFFFFFFFFFFFFF0LL) + a6 - 1))
      + (-(__int64)((a7 + 15) & 0xFFFFFFFFFFFFFFF0LL) & (((a7 + 15) & 0xFFFFFFFFFFFFFFF0LL) + 143));
  v12 = 0x7000000000200000LL;
  if ( v11 <= 0x7000000000000000LL )
  {
    _BitScanReverse64((unsigned __int64 *)&v13, 2 * v11 - 1);
    if ( (unsigned __int64)(int)v13 < 0xF )
      LOBYTE(v13) = 15;
    v12 = (((v11 + (1LL << ((unsigned __int8)v13 - 3)) - 1) & -(1LL << ((unsigned __int8)v13 - 3))) + 0x1FFFFF)
        & 0xFFFFFFFFFFE00000LL;
  }
  v14 = (unsigned int)*a4;
  v63 = 1;
  v64[0] = 1;
  v15 = (unsigned int)(v14 + 1);
  if ( (unsigned int)v15 >= 0xC6 )
    v15 = v14;
  v16 = (qword_5060180[v15] + 0x1FFFFFLL) & 0xFFFFFFFFFFE00000LL;
  if ( v16 < v12 )
    v16 = v12;
  if ( a3[1] == &off_49E8020 )
  {
    v21 = (_QWORD *)sub_13468F0(0, v16, 0x200000, &v63, v64);
    goto LABEL_16;
  }
  v17 = a3[1];
  if ( v17 == &off_49E8020 )
  {
    v32 = sub_1340EA0((_DWORD)a1, 0, v16, 0x200000, (unsigned int)&v63, (unsigned int)v64, *(unsigned int *)a3);
    v19 = v46;
    v21 = (_QWORD *)v32;
    goto LABEL_16;
  }
  if ( !a1 )
  {
    if ( __readfsbyte(0xFFFFF8C8) )
    {
      v51 = a3;
      mutex = (pthread_mutex_t *)a3[1];
      v59 = __readfsqword(0);
      v43 = sub_1313D30(v59 - 2664, 0);
      v17 = (__int64 (__fastcall **)(int, int, int, int, int, int, int))mutex;
      a3 = v51;
      ++*(_BYTE *)(v43 + 1);
      v40 = (_BYTE *)v43;
      if ( *(_BYTE *)(v43 + 816) )
      {
        v21 = (_QWORD *)((__int64 (__fastcall *)(pthread_mutex_t *, _QWORD, unsigned __int64, __int64, char *, _BYTE *, _QWORD))mutex->__align)(
                          mutex,
                          0,
                          v16,
                          0x200000,
                          &v63,
                          v64,
                          *(unsigned int *)v51);
LABEL_53:
        v20 = (_BYTE *)(v59 - 2664);
        if ( __readfsbyte(0xFFFFF8C8) )
        {
          v44 = v59 - 2664;
          v61 = v21;
          v45 = sub_1313D30(v44, 0);
          v21 = v61;
          v20 = (_BYTE *)v45;
        }
        goto LABEL_14;
      }
    }
    else
    {
      __addfsbyte(0xFFFFF599, 1u);
      v59 = __readfsqword(0);
      v40 = (_BYTE *)(v59 - 2664);
    }
    v49 = a3;
    mutexa = (pthread_mutex_t *)v17;
    sub_1313A40(v40);
    v41 = ((__int64 (__fastcall *)(pthread_mutex_t *, _QWORD, unsigned __int64, __int64, char *, _BYTE *, _QWORD))mutexa->__align)(
            mutexa,
            0,
            v16,
            0x200000,
            &v63,
            v64,
            *(unsigned int *)v49);
    v19 = v47;
    v21 = (_QWORD *)v41;
    goto LABEL_53;
  }
  ++a1[1];
  if ( a1[816] )
  {
    v18 = (*v17)((int)v17, 0, v16, 0x200000, (int)&v63, (int)v64, *(_DWORD *)a3);
    v20 = a1;
    v21 = (_QWORD *)v18;
    v22 = v48;
  }
  else
  {
    v50 = a3;
    v60 = (__int64 (__fastcall **)(_QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD))v17;
    sub_1313A40(a1);
    v42 = (*v60)(v60, 0, v16, 0x200000, &v63, v64, *(unsigned int *)v50);
    v20 = a1;
    v21 = (_QWORD *)v42;
  }
LABEL_14:
  if ( v20[1]-- == 1 )
  {
    v58 = v21;
    sub_1313A40(v20);
    v21 = v58;
  }
LABEL_16:
  if ( !v21 )
    return v21;
  if ( dword_4F96B94 && !unk_505F9C8 )
  {
    if ( dword_4F96B94 == 2 )
    {
      v62 = v21;
      sub_130CDB0(v21, v16, unk_505F9C8, v19, v21, v22);
      v21 = v62;
      goto LABEL_22;
    }
    if ( !a2 || dword_4F96B94 != 1 )
      goto LABEL_22;
    v54 = v21;
    v33 = pthread_mutex_trylock((pthread_mutex_t *)(a2 + 96));
    v37 = v54;
    if ( v33 )
    {
      sub_130AD90(a2 + 32);
      *(_BYTE *)(a2 + 136) = 1;
      v37 = v54;
    }
    ++*(_QWORD *)(a2 + 88);
    if ( a1 != *(_BYTE **)(a2 + 80) )
    {
      ++*(_QWORD *)(a2 + 72);
      *(_QWORD *)(a2 + 80) = a1;
    }
    if ( *(_BYTE *)(a2 + 144) )
      goto LABEL_45;
    v38 = *(_QWORD **)(a2 + 160);
    v39 = v38[1];
    if ( *(_DWORD *)a2 )
    {
      if ( v39 )
      {
        v34 = 2;
        do
        {
          v39 = *(_QWORD *)(v39 + 8);
          ++v34;
        }
        while ( v39 );
        if ( v34 != 2 )
          goto LABEL_46;
      }
    }
    else
    {
      if ( !v39 )
        goto LABEL_46;
      v34 = 2;
      do
      {
        v39 = *(_QWORD *)(v39 + 8);
        ++v34;
      }
      while ( v39 );
      if ( v34 != 5 )
        goto LABEL_46;
    }
    *(_BYTE *)(a2 + 144) = 1;
    do
    {
      v55 = v37;
      sub_130CDB0(v38, *v38, v34, v35, v37, v36);
      v37 = v55;
      *(_QWORD *)(a2 + 3904) += (unsigned __int64)(*v38 + 0x1FFFFFLL - v38[4]) >> 21;
      v38 = (_QWORD *)v38[1];
    }
    while ( v38 );
    if ( *(_BYTE *)(a2 + 144) )
    {
LABEL_45:
      v56 = v37;
      sub_130CDB0(v37, v16, v34, v35, v37, v36);
      v37 = v56;
    }
LABEL_46:
    v57 = v37;
    *(_BYTE *)(a2 + 136) = 0;
    pthread_mutex_unlock((pthread_mutex_t *)(a2 + 96));
    v21 = v57;
  }
LABEL_22:
  if ( v16 > 0x7000000000000000LL )
  {
    v28 = 199;
  }
  else
  {
    _BitScanReverse64(&v24, v16);
    v25 = v24 - ((((v16 - 1) & v16) == 0) - 1);
    if ( v25 < 0xE )
      v25 = 14;
    v26 = v25 - 3;
    v27 = v25 - 14;
    if ( !v27 )
      v26 = 12;
    v28 = (((v16 - 1) >> v26) & 3) + 4 * v27;
  }
  *a4 = v28;
  *v21 = v16;
  v21[1] = 0;
  v29 = (*a5)++;
  v21[6] = v29;
  v30 = v21[2];
  v21[3] = v21 + 18;
  v21[4] = v16 - 144;
  v21[2] = v30 & 0xFFFFFFFFF0000000LL | 0xE80AFFF;
  return v21;
}
