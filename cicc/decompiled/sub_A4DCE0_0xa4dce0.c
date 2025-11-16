// Function: sub_A4DCE0
// Address: 0xa4dce0
//
__int64 *__fastcall sub_A4DCE0(__int64 *a1, __int64 a2, int a3, _DWORD *a4)
{
  char *v5; // r12
  unsigned __int64 v7; // rdi
  __int64 v8; // rax
  unsigned __int64 v9; // rcx
  __int64 v10; // r9
  __int64 v11; // rsi
  __int64 v12; // rax
  _QWORD *v13; // rcx
  _QWORD *v14; // rcx
  _QWORD *v15; // rcx
  __int64 v16; // r15
  __int64 v17; // r12
  volatile signed __int32 *v18; // r13
  signed __int32 v19; // eax
  signed __int32 v20; // eax
  _QWORD *v21; // rax
  __int64 *v22; // rdi
  __int64 v23; // rcx
  __int64 v24; // r10
  __int64 v25; // rax
  _DWORD *v26; // rsi
  _DWORD *v27; // rax
  _DWORD *v28; // r9
  char *v29; // rsi
  char *v30; // rdx
  __int64 v31; // r9
  __int64 v32; // rsi
  __int64 v33; // rax
  __int64 v34; // r8
  __int64 v35; // rdx
  char v36; // al
  unsigned int v37; // eax
  __int64 v38; // rcx
  __int64 v39; // r8
  int v40; // edx
  char v41; // al
  __int64 v42; // rsi
  __int64 v43; // rcx
  __int64 v44; // rax
  __int64 v45; // rbx
  __int64 v46; // rax
  __m128i si128; // xmm0
  __int64 v48; // r13
  __int64 v50; // rax
  unsigned __int64 v51; // r9
  __int64 v52; // r12
  unsigned __int64 v53; // rax
  bool v54; // cf
  unsigned __int64 v55; // r9
  __int64 v56; // r12
  __int64 v57; // rax
  _QWORD *v58; // r13
  __int64 v59; // r15
  _QWORD *v60; // rax
  _QWORD *v61; // r9
  __int64 v62; // r10
  __int64 v63; // r10
  _QWORD *v64; // rax
  __int64 v65; // rsi
  __int64 v66; // r9
  _QWORD *v67; // r12
  __int64 v68; // rsi
  __int64 v69; // rsi
  __int64 v70; // rax
  volatile signed __int32 *v71; // r14
  signed __int32 v72; // esi
  unsigned __int64 v73; // rdi
  __int64 v74; // rdi
  char *v75; // r12
  __int64 v76; // rax
  __int64 v77; // rax
  __m128i v78; // xmm0
  __int64 v79; // [rsp+0h] [rbp-F0h]
  char *v80; // [rsp+8h] [rbp-E8h]
  __int64 v81; // [rsp+8h] [rbp-E8h]
  __int64 v84; // [rsp+28h] [rbp-C8h]
  __int64 v85; // [rsp+30h] [rbp-C0h] BYREF
  char v86; // [rsp+38h] [rbp-B8h]
  _QWORD v87[2]; // [rsp+40h] [rbp-B0h] BYREF
  _QWORD v88[2]; // [rsp+50h] [rbp-A0h] BYREF
  void *v89; // [rsp+60h] [rbp-90h] BYREF
  const char *v90; // [rsp+68h] [rbp-88h]
  int v91; // [rsp+70h] [rbp-80h]
  __int64 v92; // [rsp+78h] [rbp-78h]
  __int64 *v93; // [rsp+80h] [rbp-70h] BYREF
  __int64 v94; // [rsp+88h] [rbp-68h]
  __int64 v95; // [rsp+90h] [rbp-60h] BYREF
  __int64 v96; // [rsp+98h] [rbp-58h]
  __int64 v97; // [rsp+A0h] [rbp-50h]
  __int64 v98; // [rsp+A8h] [rbp-48h]
  _QWORD *v99; // [rsp+B0h] [rbp-40h]

  v5 = (char *)&v93;
  v7 = *(unsigned int *)(a2 + 76);
  LODWORD(v93) = *(_DWORD *)(a2 + 36);
  v8 = *(unsigned int *)(a2 + 72);
  v94 = 0;
  v9 = *(_QWORD *)(a2 + 64);
  v10 = v8 + 1;
  v95 = 0;
  v11 = v8;
  v96 = 0;
  if ( v8 + 1 > v7 )
  {
    v74 = a2 + 64;
    if ( v9 > (unsigned __int64)&v93 || (unsigned __int64)&v93 >= v9 + 32 * v8 )
    {
      sub_9D04A0(v74, v10);
      v8 = *(unsigned int *)(a2 + 72);
      v9 = *(_QWORD *)(a2 + 64);
      v11 = v8;
    }
    else
    {
      v75 = (char *)&v93 - v9;
      sub_9D04A0(v74, v10);
      v9 = *(_QWORD *)(a2 + 64);
      v8 = *(unsigned int *)(a2 + 72);
      v5 = &v75[v9];
      v11 = v8;
    }
  }
  v12 = v9 + 32 * v8;
  if ( v12 )
  {
    *(_DWORD *)v12 = *(_DWORD *)v5;
    v13 = (_QWORD *)*((_QWORD *)v5 + 1);
    *((_QWORD *)v5 + 1) = 0;
    *(_QWORD *)(v12 + 8) = v13;
    v14 = (_QWORD *)*((_QWORD *)v5 + 2);
    *((_QWORD *)v5 + 2) = 0;
    *(_QWORD *)(v12 + 16) = v14;
    v15 = (_QWORD *)*((_QWORD *)v5 + 3);
    *((_QWORD *)v5 + 3) = 0;
    *(_QWORD *)(v12 + 24) = v15;
    v16 = v95;
    v17 = v94;
    ++*(_DWORD *)(a2 + 72);
    if ( v16 != v17 )
    {
      do
      {
        while ( 1 )
        {
          v18 = *(volatile signed __int32 **)(v17 + 8);
          if ( v18 )
          {
            if ( &_pthread_key_create )
            {
              v19 = _InterlockedExchangeAdd(v18 + 2, 0xFFFFFFFF);
            }
            else
            {
              v19 = *((_DWORD *)v18 + 2);
              *((_DWORD *)v18 + 2) = v19 - 1;
            }
            if ( v19 == 1 )
            {
              (*(void (__fastcall **)(volatile signed __int32 *, __int64))(*(_QWORD *)v18 + 16LL))(v18, v11);
              if ( &_pthread_key_create )
              {
                v20 = _InterlockedExchangeAdd(v18 + 3, 0xFFFFFFFF);
              }
              else
              {
                v20 = *((_DWORD *)v18 + 3);
                *((_DWORD *)v18 + 3) = v20 - 1;
              }
              if ( v20 == 1 )
                break;
            }
          }
          v17 += 16;
          if ( v17 == v16 )
            goto LABEL_14;
        }
        v17 += 16;
        (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v18 + 24LL))(v18);
      }
      while ( v17 != v16 );
LABEL_14:
      v17 = v94;
    }
    if ( v17 )
      j_j___libc_free_0(v17, v96 - v17);
    v9 = *(_QWORD *)(a2 + 64);
  }
  else
  {
    *(_DWORD *)(a2 + 72) = v11 + 1;
  }
  v21 = (_QWORD *)(v9 + 32LL * *(unsigned int *)(a2 + 72) - 32);
  v22 = (__int64 *)v21[1];
  v23 = v21[2];
  v21[1] = *(_QWORD *)(a2 + 40);
  v24 = v21[3];
  v21[2] = *(_QWORD *)(a2 + 48);
  v21[3] = *(_QWORD *)(a2 + 56);
  v25 = *(_QWORD *)(a2 + 336);
  *(_QWORD *)(a2 + 40) = v22;
  *(_QWORD *)(a2 + 48) = v23;
  *(_QWORD *)(a2 + 56) = v24;
  if ( v25 )
  {
    v26 = *(_DWORD **)(v25 + 8);
    v27 = *(_DWORD **)v25;
    if ( v26 != v27 )
    {
      v28 = v26 - 22;
      if ( a3 != *(v26 - 22) )
      {
        while ( 1 )
        {
          v28 = v27;
          if ( a3 == *v27 )
            break;
          v27 += 22;
          if ( v26 == v27 )
            goto LABEL_33;
        }
      }
      v29 = (char *)*((_QWORD *)v28 + 2);
      v30 = (char *)*((_QWORD *)v28 + 1);
      if ( v29 != v30 )
      {
        v31 = v29 - v30;
        if ( v24 - v23 >= (unsigned __int64)(v29 - v30) )
        {
          v32 = v23 + v31;
          do
          {
            if ( v23 )
            {
              *(_QWORD *)v23 = *(_QWORD *)v30;
              v33 = *((_QWORD *)v30 + 1);
              *(_QWORD *)(v23 + 8) = v33;
              if ( v33 )
              {
                if ( &_pthread_key_create )
                  _InterlockedAdd((volatile signed __int32 *)(v33 + 8), 1u);
                else
                  ++*(_DWORD *)(v33 + 8);
              }
            }
            v23 += 16;
            v30 += 16;
          }
          while ( v23 != v32 );
          *(_QWORD *)(a2 + 48) += v31;
          goto LABEL_33;
        }
        v51 = v31 >> 4;
        v52 = 0x7FFFFFFFFFFFFFFLL;
        v53 = (v23 - (__int64)v22) >> 4;
        if ( v51 > 0x7FFFFFFFFFFFFFFLL - v53 )
          sub_4262D8((__int64)"vector::_M_range_insert");
        if ( v51 < v53 )
          v51 = (v23 - (__int64)v22) >> 4;
        v54 = __CFADD__(v53, v51);
        v55 = v53 + v51;
        if ( v54 )
        {
          v56 = 0x7FFFFFFFFFFFFFF0LL;
        }
        else
        {
          if ( !v55 )
          {
            v59 = 0;
            v58 = 0;
            goto LABEL_69;
          }
          if ( v55 <= 0x7FFFFFFFFFFFFFFLL )
            v52 = v55;
          v56 = 16 * v52;
        }
        v79 = v23;
        v80 = v30;
        v57 = sub_22077B0(v56);
        v22 = *(__int64 **)(a2 + 40);
        v30 = v80;
        v23 = v79;
        v58 = (_QWORD *)v57;
        v59 = v57 + v56;
LABEL_69:
        if ( (__int64 *)v23 == v22 )
        {
          v61 = v58;
        }
        else
        {
          v60 = v58;
          v61 = (_QWORD *)((char *)v58 + v23 - (_QWORD)v22);
          do
          {
            if ( v60 )
            {
              v62 = *v22;
              v60[1] = 0;
              *v60 = v62;
              v63 = v22[1];
              v22[1] = 0;
              v60[1] = v63;
              *v22 = 0;
            }
            v60 += 2;
            v22 += 2;
          }
          while ( v61 != v60 );
        }
        v64 = (_QWORD *)((char *)v61 + v29 - v30);
        do
        {
          if ( v61 )
          {
            *v61 = *(_QWORD *)v30;
            v65 = *((_QWORD *)v30 + 1);
            v61[1] = v65;
            if ( v65 )
            {
              if ( &_pthread_key_create )
                _InterlockedAdd((volatile signed __int32 *)(v65 + 8), 1u);
              else
                ++*(_DWORD *)(v65 + 8);
            }
          }
          v61 += 2;
          v30 += 16;
        }
        while ( v64 != v61 );
        v66 = *(_QWORD *)(a2 + 48);
        if ( v23 == v66 )
        {
          v67 = v64;
        }
        else
        {
          v67 = (_QWORD *)((char *)v64 + v66 - v23);
          do
          {
            if ( v64 )
            {
              v68 = *(_QWORD *)v23;
              v64[1] = 0;
              *v64 = v68;
              v69 = *(_QWORD *)(v23 + 8);
              *(_QWORD *)(v23 + 8) = 0;
              v64[1] = v69;
              *(_QWORD *)v23 = 0;
            }
            v64 += 2;
            v23 += 16;
          }
          while ( v64 != v67 );
          v66 = *(_QWORD *)(a2 + 48);
        }
        v70 = *(_QWORD *)(a2 + 40);
        if ( v70 != v66 )
        {
          do
          {
            v71 = *(volatile signed __int32 **)(v70 + 8);
            if ( v71 )
            {
              v23 = (__int64)&_pthread_key_create;
              if ( &_pthread_key_create )
              {
                v72 = _InterlockedExchangeAdd(v71 + 2, 0xFFFFFFFF);
              }
              else
              {
                v72 = *((_DWORD *)v71 + 2);
                *((_DWORD *)v71 + 2) = v72 - 1;
              }
              if ( v72 == 1 )
              {
                v81 = v70;
                v84 = v66;
                (*(void (__fastcall **)(volatile signed __int32 *, _QWORD, char *))(*(_QWORD *)v71 + 16LL))(
                  v71,
                  *(_QWORD *)v71,
                  v30);
                v66 = v84;
                v70 = v81;
                if ( &_pthread_key_create )
                {
                  v23 = (unsigned int)_InterlockedExchangeAdd(v71 + 3, 0xFFFFFFFF);
                }
                else
                {
                  v23 = *((unsigned int *)v71 + 3);
                  *((_DWORD *)v71 + 3) = v23 - 1;
                }
                if ( (_DWORD)v23 == 1 )
                {
                  (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v71 + 24LL))(v71);
                  v70 = v81;
                  v66 = v84;
                }
              }
            }
            v70 += 16;
          }
          while ( v70 != v66 );
          v66 = *(_QWORD *)(a2 + 40);
        }
        if ( v66 )
          j_j___libc_free_0(v66, *(_QWORD *)(a2 + 56) - v66);
        *(_QWORD *)(a2 + 40) = v58;
        *(_QWORD *)(a2 + 48) = v67;
        *(_QWORD *)(a2 + 56) = v59;
      }
    }
  }
LABEL_33:
  sub_9CE2D0((__int64)&v85, a2, 4, v23);
  v35 = v86 & 1;
  v36 = (2 * v35) | v86 & 0xFD;
  v86 = v36;
  if ( (_BYTE)v35 )
  {
    v86 = v36 & 0xFD;
    v50 = v85;
    v85 = 0;
    *a1 = v50 | 1;
LABEL_56:
    if ( v85 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v85 + 8LL))(v85);
    return a1;
  }
  v37 = v85;
  *(_DWORD *)(a2 + 36) = v85;
  if ( v37 > 0x20 )
  {
    v48 = sub_2241E50(&v85, a2, v35, (unsigned int)(2 * v35), v34);
    v98 = 0x100000000LL;
    v99 = v87;
    v93 = (__int64 *)&unk_49DD210;
    v87[0] = v88;
    v87[1] = 0;
    LOBYTE(v88[0]) = 0;
    v94 = 0;
    v95 = 0;
    v96 = 0;
    v97 = 0;
    sub_CB5980(&v93, 0, 0, 0);
    v90 = "can't read more than %zu at a time, trying to read %u";
    v92 = 32;
    v89 = &unk_49D99E0;
    v91 = *(_DWORD *)(a2 + 36);
    sub_CB6620(&v93, &v89);
    v93 = (__int64 *)&unk_49DD210;
    sub_CB5840(&v93);
    sub_A4AB40(a1, (__int64)v87, 0x54u, v48);
    if ( (_QWORD *)v87[0] != v88 )
      j_j___libc_free_0(v87[0], v88[0] + 1LL);
  }
  else
  {
    v38 = *(unsigned int *)(a2 + 32);
    if ( (unsigned int)v38 > 0x1F )
    {
      v38 = (unsigned int)(v38 - 32);
      *(_DWORD *)(a2 + 32) = 32;
      *(_QWORD *)(a2 + 24) >>= v38;
    }
    else
    {
      *(_DWORD *)(a2 + 32) = 0;
    }
    sub_9C66D0((__int64)&v89, a2, 32, v38);
    v40 = (unsigned __int8)v90 & 1;
    v41 = (2 * v40) | (unsigned __int8)v90 & 0xFD;
    LOBYTE(v90) = v41;
    if ( (_BYTE)v40 )
    {
      v73 = (unsigned __int64)v89;
      v89 = 0;
      LOBYTE(v90) = v41 & 0xFD;
      *a1 = v73 | 1;
    }
    else
    {
      if ( a4 )
        *a4 = (_DWORD)v89;
      v42 = *(unsigned int *)(a2 + 36);
      if ( (_DWORD)v42 )
      {
        v43 = *(unsigned int *)(a2 + 32);
        if ( (_DWORD)v43 || *(_QWORD *)(a2 + 8) > *(_QWORD *)(a2 + 16) )
        {
          *a1 = 1;
          goto LABEL_51;
        }
        v44 = sub_2241E50(&v89, v42, a4, v43, v39);
        v93 = &v95;
        v45 = v44;
        v87[0] = 47;
        v46 = sub_22409D0(&v93, v87, 0);
        v93 = (__int64 *)v46;
        v95 = v87[0];
        *(__m128i *)v46 = _mm_load_si128((const __m128i *)&xmmword_3F23240);
        si128 = _mm_load_si128((const __m128i *)&xmmword_3F23250);
        qmemcpy((void *)(v46 + 32), "t end of stream", 15);
        *(__m128i *)(v46 + 16) = si128;
      }
      else
      {
        v76 = sub_2241E50(&v89, v42, a4, (unsigned int)(2 * v40), v39);
        v93 = &v95;
        v45 = v76;
        v87[0] = 45;
        v77 = sub_22409D0(&v93, v87, 0);
        v93 = (__int64 *)v77;
        v95 = v87[0];
        *(__m128i *)v77 = _mm_load_si128((const __m128i *)&xmmword_3F23220);
        v78 = _mm_load_si128((const __m128i *)&xmmword_3F23230);
        qmemcpy((void *)(v77 + 32), "ode size is 0", 13);
        *(__m128i *)(v77 + 16) = v78;
      }
      v94 = v87[0];
      *((_BYTE *)v93 + v87[0]) = 0;
      sub_C63F00(a1, &v93, 84, v45);
      if ( v93 != &v95 )
        j_j___libc_free_0(v93, v95 + 1);
      if ( ((unsigned __int8)v90 & 2) != 0 )
        sub_9CDF70(&v89);
      if ( ((unsigned __int8)v90 & 1) != 0 && v89 )
        (*(void (__fastcall **)(void *))(*(_QWORD *)v89 + 8LL))(v89);
    }
  }
LABEL_51:
  if ( (v86 & 2) != 0 )
    sub_9CE230(&v85);
  if ( (v86 & 1) != 0 )
    goto LABEL_56;
  return a1;
}
