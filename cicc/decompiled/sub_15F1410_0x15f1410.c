// Function: sub_15F1410
// Address: 0x15f1410
//
_QWORD *__fastcall sub_15F1410(_QWORD *a1, _BYTE *a2, __int64 a3)
{
  _BYTE *v4; // r14
  __int64 v5; // r8
  size_t v6; // r9
  char v7; // dl
  _BYTE *v8; // rax
  _BYTE *v9; // r12
  unsigned int v10; // eax
  __int64 v11; // rdx
  unsigned __int64 *v12; // rbx
  __int64 v13; // rax
  _BYTE *v14; // rbx
  unsigned __int64 v15; // r12
  __int64 v16; // r13
  unsigned __int64 v17; // r15
  _QWORD *v18; // r13
  __int64 v19; // rbx
  _QWORD *v20; // r12
  __int64 v21; // rdx
  __int64 v23; // r15
  __int64 v24; // rdx
  unsigned __int64 v25; // r14
  unsigned __int64 v26; // r12
  __int64 v27; // rbx
  unsigned __int64 v28; // r13
  _QWORD *v29; // rbx
  unsigned __int64 v30; // r12
  _QWORD *v31; // rbx
  _BYTE *v32; // rbx
  unsigned __int64 v33; // r12
  __int64 v34; // r13
  unsigned __int64 v35; // r14
  _QWORD *v36; // r13
  __int64 v37; // rbx
  _QWORD *v38; // r12
  __int64 v39; // rax
  __int64 v40; // r15
  __int64 v41; // rdx
  unsigned __int64 v42; // r14
  unsigned __int64 v43; // r12
  __int64 v44; // rbx
  unsigned __int64 v45; // r13
  _QWORD *v46; // rbx
  unsigned __int64 v47; // r12
  _QWORD *v48; // rbx
  __int64 v49; // [rsp+10h] [rbp-140h]
  __int64 v50; // [rsp+40h] [rbp-110h]
  __int64 v51; // [rsp+40h] [rbp-110h]
  __int64 v52; // [rsp+40h] [rbp-110h]
  _DWORD *v53; // [rsp+48h] [rbp-108h]
  char v54; // [rsp+5Fh] [rbp-F1h] BYREF
  unsigned __int64 v55; // [rsp+60h] [rbp-F0h] BYREF
  unsigned __int64 v56; // [rsp+68h] [rbp-E8h]
  _BYTE *v57; // [rsp+70h] [rbp-E0h] BYREF
  __int64 v58; // [rsp+78h] [rbp-D8h]
  _BYTE v59[32]; // [rsp+80h] [rbp-D0h] BYREF
  _BYTE *v60; // [rsp+A0h] [rbp-B0h] BYREF
  __int64 v61; // [rsp+A8h] [rbp-A8h]
  _BYTE v62[160]; // [rsp+B0h] [rbp-A0h] BYREF

  v4 = a2;
  if ( !qword_4F9E240 )
    sub_16C1EA0(&qword_4F9E240, sub_12B9A60, sub_12B9AC0);
  v49 = qword_4F9E240;
  sub_16C30C0(qword_4F9E240);
  if ( !qword_4F9E260 )
    sub_16C1EA0(&qword_4F9E260, sub_15EA640, sub_15EA740);
  v53 = (_DWORD *)sub_15F09F0(qword_4F9E260, a2, a3);
  if ( v7 )
  {
    v50 = (__int64)&a2[a3];
    if ( a2 != &a2[a3] )
    {
      while ( 1 )
      {
        v55 = 0xFFFFFFFF00000000LL;
        v56 = 0;
        v57 = v59;
        v58 = 0x100000000LL;
        v54 = 44;
        v60 = v62;
        v61 = 0x200000000LL;
        v8 = sub_15EA350(v4, v50, &v54);
        v9 = v8;
        if ( v8 == v4 || (unsigned __int8)sub_15EBA30((__int64)&v55, v4, v8 - v4, (unsigned __int64)v53, v5, v6) )
          break;
        v10 = v53[2];
        if ( v10 >= v53[3] )
        {
          sub_15ECA10((__int64)v53, 0);
          v10 = v53[2];
        }
        v11 = v10;
        v12 = (unsigned __int64 *)(*(_QWORD *)v53 + 192LL * v10);
        if ( v12 )
        {
          *v12 = v55;
          v12[1] = v56;
          v12[2] = (unsigned __int64)(v12 + 4);
          v12[3] = 0x100000000LL;
          if ( (_DWORD)v58 )
            sub_15EB610((__int64)(v12 + 2), (__int64 *)&v57, v10, (__int64)v53, v5, v6);
          v12[8] = (unsigned __int64)(v12 + 10);
          v12[9] = 0x200000000LL;
          if ( (_DWORD)v61 )
            sub_15EC730((__int64)(v12 + 8), (__int64)&v60, v11, (unsigned int)v61, v5, v6);
          v10 = v53[2];
        }
        v13 = v10 + 1;
        v53[2] = v13;
        if ( (_BYTE *)v50 == v9 )
        {
          v4 = (_BYTE *)v50;
        }
        else
        {
          v4 = v9 + 1;
          if ( v9 + 1 == (_BYTE *)v50 )
          {
            v39 = *(_QWORD *)v53 + 192 * v13;
            if ( *(_QWORD *)v53 != v39 )
            {
              v52 = *(_QWORD *)v53;
              v40 = v39;
              do
              {
                v41 = *(unsigned int *)(v40 - 120);
                v42 = *(_QWORD *)(v40 - 128);
                v40 -= 192;
                v43 = v42 + 56 * v41;
                if ( v42 != v43 )
                {
                  do
                  {
                    v44 = *(unsigned int *)(v43 - 40);
                    v45 = *(_QWORD *)(v43 - 48);
                    v43 -= 56LL;
                    v46 = (_QWORD *)(v45 + 32 * v44);
                    if ( (_QWORD *)v45 != v46 )
                    {
                      do
                      {
                        v46 -= 4;
                        if ( (_QWORD *)*v46 != v46 + 2 )
                          j_j___libc_free_0(*v46, v46[2] + 1LL);
                      }
                      while ( (_QWORD *)v45 != v46 );
                      v45 = *(_QWORD *)(v43 + 8);
                    }
                    if ( v45 != v43 + 24 )
                      _libc_free(v45);
                  }
                  while ( v42 != v43 );
                  v42 = *(_QWORD *)(v40 + 64);
                }
                if ( v42 != v40 + 80 )
                  _libc_free(v42);
                v47 = *(_QWORD *)(v40 + 16);
                v48 = (_QWORD *)(v47 + 32LL * *(unsigned int *)(v40 + 24));
                if ( (_QWORD *)v47 != v48 )
                {
                  do
                  {
                    v48 -= 4;
                    if ( (_QWORD *)*v48 != v48 + 2 )
                      j_j___libc_free_0(*v48, v48[2] + 1LL);
                  }
                  while ( (_QWORD *)v47 != v48 );
                  v47 = *(_QWORD *)(v40 + 16);
                }
                if ( v47 != v40 + 32 )
                  _libc_free(v47);
              }
              while ( v52 != v40 );
            }
            goto LABEL_65;
          }
        }
        v14 = v60;
        v15 = (unsigned __int64)&v60[56 * (unsigned int)v61];
        if ( v60 != (_BYTE *)v15 )
        {
          do
          {
            v16 = *(unsigned int *)(v15 - 40);
            v17 = *(_QWORD *)(v15 - 48);
            v15 -= 56LL;
            v18 = (_QWORD *)(v17 + 32 * v16);
            if ( (_QWORD *)v17 != v18 )
            {
              do
              {
                v18 -= 4;
                if ( (_QWORD *)*v18 != v18 + 2 )
                  j_j___libc_free_0(*v18, v18[2] + 1LL);
              }
              while ( (_QWORD *)v17 != v18 );
              v17 = *(_QWORD *)(v15 + 8);
            }
            if ( v17 != v15 + 24 )
              _libc_free(v17);
          }
          while ( v14 != (_BYTE *)v15 );
          v15 = (unsigned __int64)v60;
        }
        if ( (_BYTE *)v15 != v62 )
          _libc_free(v15);
        v19 = (__int64)v57;
        v20 = &v57[32 * (unsigned int)v58];
        if ( v57 != (_BYTE *)v20 )
        {
          do
          {
            v20 -= 4;
            if ( (_QWORD *)*v20 != v20 + 2 )
              j_j___libc_free_0(*v20, v20[2] + 1LL);
          }
          while ( (_QWORD *)v19 != v20 );
          v20 = v57;
        }
        if ( v20 != (_QWORD *)v59 )
          _libc_free((unsigned __int64)v20);
        if ( v4 == (_BYTE *)v50 )
          goto LABEL_39;
      }
      if ( *(_QWORD *)v53 != *(_QWORD *)v53 + 192LL * (unsigned int)v53[2] )
      {
        v51 = *(_QWORD *)v53;
        v23 = *(_QWORD *)v53 + 192LL * (unsigned int)v53[2];
        do
        {
          v24 = *(unsigned int *)(v23 - 120);
          v25 = *(_QWORD *)(v23 - 128);
          v23 -= 192;
          v26 = v25 + 56 * v24;
          if ( v25 != v26 )
          {
            do
            {
              v27 = *(unsigned int *)(v26 - 40);
              v28 = *(_QWORD *)(v26 - 48);
              v26 -= 56LL;
              v29 = (_QWORD *)(v28 + 32 * v27);
              if ( (_QWORD *)v28 != v29 )
              {
                do
                {
                  v29 -= 4;
                  if ( (_QWORD *)*v29 != v29 + 2 )
                    j_j___libc_free_0(*v29, v29[2] + 1LL);
                }
                while ( (_QWORD *)v28 != v29 );
                v28 = *(_QWORD *)(v26 + 8);
              }
              if ( v28 != v26 + 24 )
                _libc_free(v28);
            }
            while ( v25 != v26 );
            v25 = *(_QWORD *)(v23 + 64);
          }
          if ( v25 != v23 + 80 )
            _libc_free(v25);
          v30 = *(_QWORD *)(v23 + 16);
          v31 = (_QWORD *)(v30 + 32LL * *(unsigned int *)(v23 + 24));
          if ( (_QWORD *)v30 != v31 )
          {
            do
            {
              v31 -= 4;
              if ( (_QWORD *)*v31 != v31 + 2 )
                j_j___libc_free_0(*v31, v31[2] + 1LL);
            }
            while ( (_QWORD *)v30 != v31 );
            v30 = *(_QWORD *)(v23 + 16);
          }
          if ( v30 != v23 + 32 )
            _libc_free(v30);
        }
        while ( v51 != v23 );
      }
LABEL_65:
      v53[2] = 0;
      v32 = v60;
      v33 = (unsigned __int64)&v60[56 * (unsigned int)v61];
      if ( v60 != (_BYTE *)v33 )
      {
        do
        {
          v34 = *(unsigned int *)(v33 - 40);
          v35 = *(_QWORD *)(v33 - 48);
          v33 -= 56LL;
          v36 = (_QWORD *)(v35 + 32 * v34);
          if ( (_QWORD *)v35 != v36 )
          {
            do
            {
              v36 -= 4;
              if ( (_QWORD *)*v36 != v36 + 2 )
                j_j___libc_free_0(*v36, v36[2] + 1LL);
            }
            while ( (_QWORD *)v35 != v36 );
            v35 = *(_QWORD *)(v33 + 8);
          }
          if ( v35 != v33 + 24 )
            _libc_free(v35);
        }
        while ( v32 != (_BYTE *)v33 );
        v33 = (unsigned __int64)v60;
      }
      if ( (_BYTE *)v33 != v62 )
        _libc_free(v33);
      v37 = (__int64)v57;
      v38 = &v57[32 * (unsigned int)v58];
      if ( v57 != (_BYTE *)v38 )
      {
        do
        {
          v38 -= 4;
          if ( (_QWORD *)*v38 != v38 + 2 )
            j_j___libc_free_0(*v38, v38[2] + 1LL);
        }
        while ( (_QWORD *)v37 != v38 );
        v38 = v57;
      }
      if ( v38 != (_QWORD *)v59 )
        _libc_free((unsigned __int64)v38);
    }
  }
LABEL_39:
  *a1 = a1 + 2;
  a1[1] = 0x400000000LL;
  v21 = (unsigned int)v53[2];
  if ( (_DWORD)v21 )
    sub_15ECD30((__int64)a1, v53, v21, 0x400000000LL, v5, v6);
  sub_16C30E0(v49);
  return a1;
}
