// Function: sub_1345C30
// Address: 0x1345c30
//
unsigned __int64 *__fastcall sub_1345C30(
        _BYTE *a1,
        __int64 a2,
        unsigned int *a3,
        __int64 a4,
        __int64 *a5,
        unsigned __int64 a6,
        unsigned __int64 a7,
        char a8,
        char a9)
{
  unsigned __int64 *v12; // r15
  unsigned __int64 v14; // rcx
  __int64 v15; // rax
  unsigned __int64 v16; // r13
  __int64 v17; // rax
  unsigned __int64 *v18; // rax
  int v19; // ebx
  __int64 (__fastcall **v20)(int, int, int, int, int, int, int); // r11
  __int64 v21; // rax
  _BYTE *v22; // r10
  __int64 v23; // r8
  signed __int64 v25; // rax
  unsigned __int64 *v26; // rcx
  __int64 v27; // rsi
  unsigned __int64 v28; // r9
  __int64 v29; // rax
  unsigned int v30; // eax
  unsigned __int64 *v31; // rax
  _BYTE *v32; // rdi
  __int64 v33; // rdi
  __int64 v34; // rax
  size_t v35; // rdx
  void *v36; // rdi
  __int64 v37; // rax
  __int64 v38; // [rsp+8h] [rbp-A8h]
  unsigned __int8 v39; // [rsp+10h] [rbp-A0h]
  __int64 (__fastcall **v40)(_QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD); // [rsp+10h] [rbp-A0h]
  __int64 (__fastcall **v41)(int, int, int, int, int, int, int); // [rsp+10h] [rbp-A0h]
  unsigned __int8 v42; // [rsp+18h] [rbp-98h]
  unsigned __int64 v43; // [rsp+18h] [rbp-98h]
  __int64 v44; // [rsp+18h] [rbp-98h]
  __int64 v45; // [rsp+18h] [rbp-98h]
  __int64 (__fastcall **v46)(_QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD); // [rsp+18h] [rbp-98h]
  __int64 v47; // [rsp+20h] [rbp-90h]
  pthread_mutex_t *mutex; // [rsp+30h] [rbp-80h]
  unsigned __int8 v51; // [rsp+55h] [rbp-5Bh] BYREF
  unsigned __int8 v52; // [rsp+56h] [rbp-5Ah] BYREF
  unsigned __int8 v53; // [rsp+57h] [rbp-59h] BYREF
  unsigned __int64 *v54; // [rsp+58h] [rbp-58h] BYREF
  unsigned __int64 *v55; // [rsp+60h] [rbp-50h] BYREF
  unsigned __int64 *v56; // [rsp+68h] [rbp-48h] BYREF
  unsigned __int64 *v57; // [rsp+70h] [rbp-40h] BYREF
  unsigned __int64 *v58; // [rsp+78h] [rbp-38h] BYREF

  v51 = 1;
  mutex = (pthread_mutex_t *)(a2 + 58472);
  if ( pthread_mutex_trylock((pthread_mutex_t *)(a2 + 58472)) )
  {
    sub_130AD90(a2 + 58408);
    *(_BYTE *)(a2 + 58512) = 1;
  }
  ++*(_QWORD *)(a2 + 58464);
  if ( a1 != *(_BYTE **)(a2 + 58456) )
  {
    ++*(_QWORD *)(a2 + 58448);
    *(_QWORD *)(a2 + 58456) = a1;
  }
  v47 = a2 + 38936;
  v12 = sub_1345880(a1, (__int64 *)a2, a3, a2 + 38936, a5, a6, a7, a8, (char *)&v51, 1, a9);
  if ( !v12 )
  {
    if ( !unk_4C6F2C8 || a5 || a9 == 1 )
    {
      *(_BYTE *)(a2 + 58512) = 0;
      pthread_mutex_unlock(mutex);
      if ( unk_4C6F2C8 )
      {
        if ( a5 )
          return v12;
        if ( a9 )
          return 0;
      }
      else
      {
        if ( a9 )
          return v12;
        if ( a5 )
          v12 = (unsigned __int64 *)((a5[1] & 0xFFFFFFFFFFFFF000LL) + (a5[2] & 0xFFFFFFFFFFFFF000LL));
      }
      return sub_1344390(a1, a2, a3, (__int64)v12, a6, a7, a8, &v51);
    }
    v14 = a6 + ((a7 + 4095) & 0xFFFFFFFFFFFFF000LL) - 4096;
    if ( a6 > v14 )
      goto LABEL_50;
    v15 = *(unsigned int *)(a2 + 58400);
    v16 = qword_5060180[v15];
    if ( v14 <= v16 )
    {
      v19 = 0;
    }
    else
    {
      v17 = (unsigned int)(v15 + 1);
      if ( (unsigned int)v17 > 0xC5 )
        goto LABEL_50;
      v18 = &qword_5060180[v17];
      v19 = 1;
      while ( 1 )
      {
        v16 = *v18;
        if ( v14 <= *v18 )
          break;
        ++v19;
        ++v18;
        if ( v19 == 198 - *(_DWORD *)(a2 + 58400) )
          goto LABEL_50;
      }
    }
    v54 = sub_1340A00(a1, *(_QWORD *)(a2 + 58392));
    if ( !v54 )
      goto LABEL_50;
    v52 = 0;
    v53 = 0;
    v20 = (__int64 (__fastcall **)(int, int, int, int, int, int, int))*((_QWORD *)a3 + 1);
    if ( v20 == &off_49E8020 )
    {
      v23 = sub_1340EA0((__int64)a1, 0, v16, 4096, (__int64)&v52, (__int64)&v53, *a3);
LABEL_25:
      if ( v23 )
      {
        v38 = v23;
        v39 = v52;
        v42 = v53;
        v25 = sub_13441B0(a2);
        v26 = v54;
        v27 = *(unsigned int *)(a2 + 58364);
        v28 = v25;
        v29 = *v54;
        v54[4] = v28;
        v26[1] = v38;
        v26[2] = v16 | v26[2] & 0xFFF;
        *v26 = ((unsigned __int64)v42 << 13) & 0xFFFFEFFFFFFFBFFFLL
             | ((unsigned __int64)v39 << 15) & 0xFFFFEFFFFFFFBFFFLL
             | v27 & 0xFFFFEFFFF0000FFFLL
             | v29 & 0xFFFFEFFFF0000000LL
             | 0x10000E800000LL;
        if ( !(unsigned __int8)sub_1341BA0((__int64)a1, *(_QWORD *)(a2 + 58384), v54, 0xE8u, 0) )
        {
          if ( (*v54 & 0x2000) != 0 )
            v51 = 1;
          v57 = 0;
          v58 = 0;
          if ( (unsigned int)sub_1343C70(a1, a2, a3, &v54, &v55, &v56, &v57, &v58, a6, a7) )
          {
            if ( v58 )
              sub_13451C0(a1, (__int64 *)a2, a3, v47, (__int64 *)v58);
            if ( v57 )
            {
              sub_1341E90((__int64)a1, *(_QWORD *)(a2 + 58384), (__int64)v57);
              sub_1343DD0(a1, a2, a3, v47, v57);
            }
            goto LABEL_50;
          }
          if ( v55 )
            sub_13451C0(a1, (__int64 *)a2, a3, v47, (__int64 *)v55);
          if ( v56 )
            sub_13451C0(a1, (__int64 *)a2, a3, v47, (__int64 *)v56);
          if ( v51
            && (*v54 & 0x2000) == 0
            && (unsigned __int8)sub_1343140(a1, a3, (__int64 *)v54, 0, v54[2] & 0xFFFFFFFFFFFFF000LL) )
          {
            sub_13451C0(a1, (__int64 *)a2, a3, v47, (__int64 *)v54);
            goto LABEL_50;
          }
          v30 = v19 + *(_DWORD *)(a2 + 58400) + 1;
          if ( v30 > *(_DWORD *)(a2 + 58404) )
            v30 = *(_DWORD *)(a2 + 58404);
          *(_DWORD *)(a2 + 58400) = v30;
          *(_BYTE *)(a2 + 58512) = 0;
          pthread_mutex_unlock(mutex);
          if ( a8 )
          {
            v31 = v54;
            if ( (*v54 & 0x8000) != 0 )
              return v31;
            v35 = v54[2] & 0xFFFFFFFFFFFFF000LL;
            v36 = (void *)(v54[1] & 0xFFFFFFFFFFFFF000LL);
            if ( *((__int64 (__fastcall ***)(int, int, int, int, int, int, int))a3 + 1) == &off_49E8020 )
              sub_1341200(v36, v35);
            else
              memset(v36, 0, v35);
          }
          v31 = v54;
          if ( !v54 )
            return sub_1344390(a1, a2, a3, (__int64)v12, a6, a7, a8, &v51);
          return v31;
        }
      }
      sub_1340AC0((__int64)a1, *(_QWORD *)(a2 + 58392), v54);
LABEL_50:
      *(_BYTE *)(a2 + 58512) = 0;
      pthread_mutex_unlock(mutex);
      return sub_1344390(a1, a2, a3, (__int64)v12, a6, a7, a8, &v51);
    }
    if ( a1 )
    {
      ++a1[1];
      if ( a1[816] )
      {
        v21 = (*v20)((int)v20, 0, v16, 4096, (int)&v52, (int)&v53, *a3);
      }
      else
      {
        v46 = (__int64 (__fastcall **)(_QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD))v20;
        sub_1313A40(a1);
        v21 = (*v46)(v46, 0, v16, 4096, &v52, &v53, *a3);
      }
      v22 = a1;
      v23 = v21;
LABEL_23:
      if ( v22[1]-- == 1 )
      {
        v45 = v23;
        sub_1313A40(v22);
        v23 = v45;
      }
      goto LABEL_25;
    }
    if ( __readfsbyte(0xFFFFF8C8) )
    {
      v41 = (__int64 (__fastcall **)(int, int, int, int, int, int, int))*((_QWORD *)a3 + 1);
      v43 = __readfsqword(0);
      v37 = sub_1313D30(v43 - 2664, 0);
      v20 = v41;
      ++*(_BYTE *)(v37 + 1);
      v32 = (_BYTE *)v37;
      if ( *(_BYTE *)(v37 + 816) )
      {
        v23 = (*v41)((int)v41, 0, v16, 4096, (int)&v52, (int)&v53, *a3);
LABEL_54:
        v22 = (_BYTE *)(v43 - 2664);
        if ( __readfsbyte(0xFFFFF8C8) )
        {
          v33 = v43 - 2664;
          v44 = v23;
          v34 = sub_1313D30(v33, 0);
          v23 = v44;
          v22 = (_BYTE *)v34;
        }
        goto LABEL_23;
      }
    }
    else
    {
      __addfsbyte(0xFFFFF599, 1u);
      v43 = __readfsqword(0);
      v32 = (_BYTE *)(v43 - 2664);
    }
    v40 = (__int64 (__fastcall **)(_QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD))v20;
    sub_1313A40(v32);
    v23 = (*v40)(v40, 0, v16, 4096, &v52, &v53, *a3);
    goto LABEL_54;
  }
  *(_BYTE *)(a2 + 58512) = 0;
  pthread_mutex_unlock(mutex);
  return v12;
}
