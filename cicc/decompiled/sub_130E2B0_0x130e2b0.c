// Function: sub_130E2B0
// Address: 0x130e2b0
//
__int64 __fastcall sub_130E2B0(
        __int64 a1,
        __int64 a2,
        unsigned __int64 a3,
        unsigned __int64 a4,
        unsigned __int8 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8)
{
  unsigned __int64 v11; // rax
  unsigned int v12; // eax
  char v13; // cl
  unsigned int v14; // eax
  __int64 v15; // rbx
  __int64 v16; // rax
  __int64 v17; // r14
  __int64 v18; // rbx
  int v19; // eax
  pthread_mutex_t *v20; // r10
  pthread_mutex_t *v21; // r11
  pthread_mutex_t *v22; // rdx
  __int64 v23; // r9
  pthread_mutex_t *v24; // r8
  pthread_mutex_t *align; // rax
  __int64 v26; // rax
  __int64 v29; // rdi
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // r8
  pthread_mutex_t *v33; // r10
  pthread_mutex_t *v34; // r11
  unsigned __int64 v35; // r15
  __int64 v36; // rax
  __int64 v37; // rax
  int v38; // eax
  pthread_mutex_t *v39; // r10
  pthread_mutex_t *v40; // r11
  pthread_mutex_t *v41; // r8
  pthread_mutex_t *v42; // rdx
  __int64 v43; // rcx
  __int64 v44; // rax
  __int64 v45; // rdx
  __int64 v46; // rsi
  __int64 v47; // rcx
  __int64 v48; // rcx
  unsigned __int64 v49; // rdx
  unsigned __int64 v50; // rcx
  unsigned __int64 v51; // rcx
  pthread_mutex_t *v52; // [rsp+10h] [rbp-60h]
  pthread_mutex_t *v53; // [rsp+18h] [rbp-58h]
  pthread_mutex_t *v54; // [rsp+18h] [rbp-58h]
  __int64 v55; // [rsp+18h] [rbp-58h]
  pthread_mutex_t *v57; // [rsp+20h] [rbp-50h]
  pthread_mutex_t *v58; // [rsp+20h] [rbp-50h]
  pthread_mutex_t *mutexc; // [rsp+28h] [rbp-48h]
  pthread_mutex_t *mutex; // [rsp+28h] [rbp-48h]
  pthread_mutex_t *mutexa; // [rsp+28h] [rbp-48h]
  pthread_mutex_t *mutexb; // [rsp+28h] [rbp-48h]
  pthread_mutex_t *mutexd; // [rsp+28h] [rbp-48h]
  pthread_mutex_t *mutexe; // [rsp+28h] [rbp-48h]
  char v65; // [rsp+37h] [rbp-39h] BYREF
  __int64 v66[7]; // [rsp+38h] [rbp-38h] BYREF

  if ( a4 > 0x1000 || a5 || !*(_QWORD *)(a2 + 64) || *(_QWORD *)(a2 + 72) < a3 )
  {
    LODWORD(a7) = (unsigned __int8)a7;
    return (**(__int64 (__fastcall ***)(__int64, _QWORD, unsigned __int64, unsigned __int64, _QWORD, _QWORD, __int64, __int64))(a2 + 56))(
             a1,
             *(_QWORD *)(a2 + 56),
             a3,
             a4,
             a5,
             0,
             a7,
             a8);
  }
  else
  {
    if ( a3 > 0x7000000000000000LL )
    {
      v15 = 4776;
    }
    else
    {
      _BitScanReverse64(&v11, a3);
      v12 = v11 - ((((a3 - 1) & a3) == 0) - 1);
      if ( v12 < 0xE )
        v12 = 14;
      v13 = v12 - 3;
      v14 = v12 - 14;
      if ( !v14 )
        v13 = 12;
      v15 = 24 * ((((a3 - 1) >> v13) & 3) + 4 * v14);
    }
    if ( a1 )
    {
      v16 = *(unsigned __int8 *)(a1 + 160);
      if ( (_BYTE)v16 == 0xFF )
      {
        v50 = 0x5851F42D4C957F2DLL * *(_QWORD *)(a1 + 112) + 0x14057B7EF767814FLL;
        *(_QWORD *)(a1 + 112) = v50;
        v51 = (*(_QWORD *)(a2 + 64) * HIDWORD(v50)) >> 32;
        *(_BYTE *)(a1 + 160) = v51;
        v16 = (unsigned __int8)v51;
      }
      v17 = *(_QWORD *)(a2 + 104) + 144 * v16;
    }
    else
    {
      v17 = *(_QWORD *)(a2 + 104);
    }
    v18 = *(_QWORD *)(v17 + 120) + v15;
    v19 = pthread_mutex_trylock((pthread_mutex_t *)(v17 + 64));
    v20 = (pthread_mutex_t *)(v17 + 64);
    v21 = (pthread_mutex_t *)(v17 + 104);
    v22 = (pthread_mutex_t *)a3;
    v23 = a8;
    if ( v19 )
    {
      sub_130AD90(v17);
      *(_BYTE *)(v17 + 104) = 1;
      v23 = a8;
      v22 = (pthread_mutex_t *)a3;
      v21 = (pthread_mutex_t *)(v17 + 104);
      v20 = (pthread_mutex_t *)(v17 + 64);
    }
    ++*(_QWORD *)(v17 + 56);
    if ( a1 != *(_QWORD *)(v17 + 48) )
    {
      ++*(_QWORD *)(v17 + 40);
      *(_QWORD *)(v17 + 48) = a1;
    }
    if ( *(_BYTE *)(v17 + 112) )
    {
      v24 = *(pthread_mutex_t **)(v18 + 16);
      if ( v24 )
      {
        align = (pthread_mutex_t *)v24[1].__align;
        *(_QWORD *)(v18 + 16) = align;
        if ( v24 == align )
        {
          *(_QWORD *)(v18 + 16) = 0;
        }
        else
        {
          *(_QWORD *)(*(&v24[1].__align + 1) + 40) = *(&align[1].__align + 1);
          v26 = *(&v24[1].__align + 1);
          *(_QWORD *)(v24[1].__align + 48) = v26;
          *(&v24[1].__align + 1) = *(_QWORD *)(v26 + 40);
          *(_QWORD *)(*(_QWORD *)(v24[1].__align + 48) + 40LL) = v24[1].__align;
          *(_QWORD *)(*(&v24[1].__align + 1) + 40) = v24;
        }
        mutexc = v24;
        *(_QWORD *)(v18 + 8) -= *(&v24->__align + 2) & 0xFFFFFFFFFFFFF000LL;
        *(_QWORD *)(v17 + 128) -= *(&v24->__align + 2) & 0xFFFFFFFFFFFFF000LL;
        v21->__size[0] = 0;
        pthread_mutex_unlock(v20);
        return (__int64)mutexc;
      }
    }
    if ( !*(_BYTE *)v18 && *(_QWORD *)(a2 + 96) )
    {
      mutex = v22;
      *(_BYTE *)v18 = 1;
      v21->__size[0] = 0;
      v53 = v21;
      v57 = v20;
      pthread_mutex_unlock(v20);
      v29 = *(_QWORD *)(a2 + 96);
      v30 = *(_QWORD *)(a2 + 56);
      v66[0] = 0;
      v65 = 0;
      v31 = (*(__int64 (__fastcall **)(__int64, __int64, pthread_mutex_t *, __int64, __int64 *, char *))(v30 + 8))(
              a1,
              v30,
              mutex,
              v29 + 1,
              v66,
              &v65);
      v32 = v66[0];
      v33 = v57;
      v34 = v53;
      v35 = v31;
      if ( v66[0] )
      {
        v36 = *(_QWORD *)(v66[0] + 40);
        v66[0] = v36;
        if ( v32 == v36 )
        {
          v66[0] = 0;
        }
        else
        {
          *(_QWORD *)(*(_QWORD *)(v32 + 48) + 40LL) = *(_QWORD *)(v36 + 48);
          v37 = *(_QWORD *)(v32 + 48);
          *(_QWORD *)(*(_QWORD *)(v32 + 40) + 48LL) = v37;
          *(_QWORD *)(v32 + 48) = *(_QWORD *)(v37 + 40);
          *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v32 + 40) + 48LL) + 40LL) = *(_QWORD *)(v32 + 40);
          *(_QWORD *)(*(_QWORD *)(v32 + 48) + 40LL) = v32;
        }
      }
      v52 = mutex;
      v54 = (pthread_mutex_t *)v32;
      v58 = v34;
      mutexa = v33;
      v38 = pthread_mutex_trylock(v33);
      v39 = mutexa;
      v40 = v58;
      v41 = v54;
      v42 = v52;
      if ( v38 )
      {
        sub_130AD90(v17);
        v40 = v58;
        v58->__size[0] = 1;
        v42 = v52;
        v41 = v54;
        v39 = mutexa;
      }
      ++*(_QWORD *)(v17 + 56);
      if ( a1 != *(_QWORD *)(v17 + 48) )
      {
        ++*(_QWORD *)(v17 + 40);
        *(_QWORD *)(v17 + 48) = a1;
      }
      *(_BYTE *)v18 = 0;
      if ( v35 <= 1 )
      {
        mutexe = v41;
        v40->__size[0] = 0;
        pthread_mutex_unlock(v39);
        return (__int64)mutexe;
      }
      else
      {
        v43 = *(_QWORD *)(v18 + 16);
        v44 = v66[0];
        v45 = (v35 - 1) * (_QWORD)v42;
        if ( v43 )
        {
          if ( v66[0] )
          {
            v46 = *(_QWORD *)(v66[0] + 48);
            v47 = *(_QWORD *)(v43 + 48);
            v66[0] = 0;
            *(_QWORD *)(v46 + 40) = v47;
            v48 = *(_QWORD *)(v44 + 48);
            *(_QWORD *)(*(_QWORD *)(v18 + 16) + 48LL) = v48;
            *(_QWORD *)(v44 + 48) = *(_QWORD *)(v48 + 40);
            *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v18 + 16) + 48LL) + 40LL) = *(_QWORD *)(v18 + 16);
            *(_QWORD *)(*(_QWORD *)(v44 + 48) + 40LL) = v44;
          }
        }
        else
        {
          *(_QWORD *)(v18 + 16) = v66[0];
          v66[0] = 0;
        }
        *(_QWORD *)(v18 + 8) += v45;
        v49 = *(_QWORD *)(v17 + 128) + v45;
        *(_QWORD *)(v17 + 128) = v49;
        mutexb = v41;
        if ( v49 <= *(_QWORD *)(a2 + 80) )
        {
          v40->__size[0] = 0;
          pthread_mutex_unlock(v39);
        }
        else
        {
          sub_130E170(a1, a2, v17);
        }
        return (__int64)mutexb;
      }
    }
    v55 = v23;
    mutexd = v22;
    v21->__size[0] = 0;
    pthread_mutex_unlock(v20);
    LODWORD(a7) = (unsigned __int8)a7;
    return (**(__int64 (__fastcall ***)(__int64, _QWORD, pthread_mutex_t *, unsigned __int64, _QWORD, _QWORD, __int64, __int64))(a2 + 56))(
             a1,
             *(_QWORD *)(a2 + 56),
             mutexd,
             a4,
             0,
             0,
             a7,
             v55);
  }
}
