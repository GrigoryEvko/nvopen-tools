// Function: sub_1685080
// Address: 0x1685080
//
_QWORD *__fastcall sub_1685080(__int64 a1, __int64 a2)
{
  unsigned __int64 v3; // rbx
  unsigned __int64 v4; // rbx
  unsigned int v5; // eax
  __int64 v6; // rsi
  unsigned int v7; // ecx
  _QWORD *v8; // rax
  unsigned __int64 v9; // rdx
  _QWORD *v10; // r14
  __int64 v11; // rdx
  __int64 v12; // rdi
  __int64 v13; // rdx
  int v14; // ecx
  int v15; // r8d
  int v16; // r9d
  __int64 v17; // r14
  char *v18; // rdi
  unsigned int v19; // ecx
  char *v20; // rdi
  unsigned __int64 v21; // r8
  unsigned __int64 v22; // r15
  int v23; // esi
  __int64 v24; // rdi
  _QWORD *v25; // rdx
  int v26; // ecx
  int v27; // r8d
  int v28; // r9d
  int v29; // eax
  _QWORD *v30; // rdx
  _QWORD *v31; // rcx
  int v32; // eax
  __int64 v33; // rdi
  __int64 v34; // rax
  __int64 v35; // rax
  int v36; // eax
  unsigned int v37; // eax
  __int64 v38; // r14
  _QWORD *v39; // r15
  unsigned __int64 v41; // rcx
  _QWORD *v42; // rax
  __int64 v43; // rax
  __int64 v44; // rdi
  int v45; // edx
  int v46; // ecx
  int v47; // r8d
  int v48; // r9d
  __int64 v49; // rax
  _QWORD *v50; // r12
  int v51; // eax
  __int64 v52; // rdx
  __int64 v53; // rax
  int v54; // eax
  __int64 v55; // rdi
  __int64 v56; // rax
  int v57; // edx
  int v58; // ecx
  int v59; // r9d
  unsigned __int64 v60; // r10
  __int64 v61; // r8
  __int64 v62; // rax
  unsigned __int64 v63; // r12
  __int64 v64; // rdi
  __int64 v65; // rax
  __int64 v66; // rdx
  int v67; // r9d
  __int64 v68; // r8
  unsigned __int64 v69; // r10
  __int64 v70; // rcx
  int v71; // eax
  _QWORD *v72; // rcx
  char *v73; // r12
  char v74; // [rsp+0h] [rbp-50h]
  unsigned __int64 v75; // [rsp+0h] [rbp-50h]
  __int64 v76; // [rsp+0h] [rbp-50h]
  __int64 v77; // [rsp+8h] [rbp-48h]
  unsigned __int64 v78; // [rsp+8h] [rbp-48h]
  _QWORD *v79; // [rsp+8h] [rbp-48h]
  __int64 v80; // [rsp+8h] [rbp-48h]
  _QWORD *v81; // [rsp+10h] [rbp-40h]
  __int64 v82; // [rsp+10h] [rbp-40h]
  __int64 v83; // [rsp+18h] [rbp-38h]
  __int64 v84; // [rsp+18h] [rbp-38h]
  __int64 v85; // [rsp+18h] [rbp-38h]
  unsigned __int64 v86; // [rsp+18h] [rbp-38h]

  if ( a1 )
  {
    sub_1684B50((pthread_mutex_t **)(a1 + 7128));
    v3 = (a2 + 7) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v3 > 0x1387 )
    {
      v4 = v3 + 32;
      while ( 1 )
      {
        v5 = sub_1683CB0(v4);
        v6 = *(unsigned int *)(a1 + 60);
        v7 = v5;
        if ( (unsigned int)v6 >= v5 )
        {
          while ( 2 )
          {
            v8 = *(_QWORD **)(32LL * v7 + a1 + 64);
            while ( v8 )
            {
              v9 = v8[2];
              v10 = v8;
              v8 = (_QWORD *)*v8;
              if ( v4 <= v9 )
              {
                v39 = v10 + 4;
                v41 = v9 - v4;
                if ( v8 )
                  v8[1] = v10[1];
                v42 = (_QWORD *)v10[1];
                if ( v42 )
                  *v42 = *v10;
                *v10 = -1;
                if ( v41 > 0x27 )
                {
                  v50 = (_QWORD *)((char *)v10 + v4);
                  v10[2] = v4;
                  v50[2] = v41;
                  v50[3] = v4;
                  *(_QWORD *)((char *)v10 + v9 + 24) = v41;
                  if ( (int)sub_1683CB0(*(_QWORD *)((char *)v10 + v4 + 16)) >= 0 )
                  {
                    v51 = sub_1683CB0(v50[2]);
                    v52 = v51 + 2LL;
                    v53 = a1 + 32LL * v51;
                    v50[1] = a1 + 32 * v52;
                    *v50 = *(_QWORD *)(v53 + 64);
                    *(_QWORD *)(v53 + 64) = v50;
                    if ( *v50 )
                      *(_QWORD *)(*v50 + 8LL) = v50;
                  }
                  v54 = *(_DWORD *)(a1 + 56);
                  if ( v54 )
                    *(_DWORD *)(a1 + 56) = v54 - 1;
                }
                v43 = sub_1684C80((unsigned __int64)v10);
                if ( v43 )
                  *(_QWORD *)(v43 + 8) -= v10[2];
                goto LABEL_26;
              }
            }
            if ( (unsigned int)v6 >= ++v7 )
              continue;
            break;
          }
        }
        if ( !(unsigned __int8)sub_1684BB0(v4) )
          goto LABEL_59;
        v83 = sub_1683C60(0);
        v12 = *(_QWORD *)(sub_1689050(0, v6, v11) + 24);
        v17 = sub_1685080(v12, 88);
        if ( !v17 )
          sub_1683C30(v12, 88, v13, v14, v15, v16, v74);
        *(_QWORD *)v17 = 0;
        v18 = (char *)((v17 + 8) & 0xFFFFFFFFFFFFFFF8LL);
        *(_QWORD *)(v17 + 80) = 0;
        v19 = (unsigned int)(v17 - (_DWORD)v18 + 88) >> 3;
        memset(v18, 0, 8LL * v19);
        v20 = &v18[8 * v19];
        v21 = *(unsigned int *)(a1 + 32);
        if ( v21 < v4 )
          v21 = v4;
        v22 = v21;
        v23 = v21 + 64;
        v24 = *(_QWORD *)(sub_1689050(v20, 88, v13) + 24);
        v25 = (_QWORD *)sub_1685080(v24, v22 + 64);
        if ( !v25 )
        {
          sub_1683C30(v24, v23, 0, v26, v27, v28, v74);
          v25 = 0;
        }
        *v25 = -1;
        v25[1] = 0;
        v25[2] = 32;
        v25[3] = 0;
        v25[6] = v22;
        v25[7] = 32;
        v77 = (__int64)v25 + v22 + 32;
        v81 = v25;
        v29 = sub_1683CB0(v22);
        v30 = v81;
        v31 = (_QWORD *)v77;
        if ( v29 >= 0 )
        {
          v74 = v77;
          v32 = sub_1683CB0(v81[6]);
          v30 = v81;
          v31 = (_QWORD *)v77;
          v33 = v32 + 2LL;
          v34 = a1 + 32LL * v32;
          v81[5] = a1 + 32 * v33;
          v81[4] = *(_QWORD *)(v34 + 64);
          *(_QWORD *)(v34 + 64) = v81 + 4;
          v35 = v81[4];
          if ( v35 )
            *(_QWORD *)(v35 + 8) = v81 + 4;
        }
        *v31 = -1;
        v31[1] = 0;
        v31[2] = 32;
        v31[3] = v22;
        *(_QWORD *)(v17 + 8) = v22;
        *(_QWORD *)(v17 + 16) = v22;
        *(_QWORD *)(v17 + 24) = a1;
        *(_QWORD *)(v17 + 32) = v30;
        *(_BYTE *)(v17 + 40) = 0;
        _InterlockedAdd(&dword_4F9F340, 1u);
        v36 = dword_4F9F340;
        *(_QWORD *)(v17 + 48) = v31;
        *(_DWORD *)(v17 + 44) = v36;
        *(_QWORD *)v17 = *(_QWORD *)(a1 + 48);
        *(_QWORD *)(a1 + 48) = v17;
        v37 = sub_1683CB0(v22);
        if ( *(_DWORD *)(a1 + 60) >= v37 )
          v37 = *(_DWORD *)(a1 + 60);
        *(_DWORD *)(a1 + 60) = v37;
        sub_1684D30(*(_QWORD *)(v17 + 32) >> 3, *(_QWORD *)(v17 + 16) >> 3, v17);
        sub_1683C60(v83);
        sub_1684B50(&qword_4F9F360);
        --dword_4F9F34C;
        j__pthread_mutex_unlock(qword_4F9F360);
      }
    }
    if ( v3 < 0x10 )
      v3 = 16;
    v38 = (unsigned int)(v3 >> 3);
    v39 = *(_QWORD **)(a1 + 8 * v38 + 2128);
    if ( !v39 )
    {
      if ( !(unsigned __int8)sub_1684BB0(*(unsigned int *)(a1 + 32)) )
      {
LABEL_59:
        v39 = 0;
        goto LABEL_26;
      }
      v82 = sub_1683C60(0);
      v55 = *(_QWORD *)(((__int64 (*)(void))sub_1689050)() + 24);
      v56 = sub_1685080(v55, 56);
      v60 = v3 >> 3;
      v61 = v56;
      if ( !v56 )
      {
        sub_1683C30(v55, 56, v57, v58, 0, v59, v74);
        v61 = 0;
        v60 = v3 >> 3;
      }
      *(_QWORD *)(v61 + 48) = 0;
      *(_OWORD *)v61 = 0;
      *(_OWORD *)(v61 + 16) = 0;
      *(_OWORD *)(v61 + 32) = 0;
      v62 = *(unsigned int *)(a1 + 32);
      v78 = v60;
      v84 = v61;
      v63 = v3 * ((v3 + v62 - 1) / v3);
      v64 = *(_QWORD *)(sub_1689050(v55, 56, (v3 + v62 - 1) % v3) + 24);
      v65 = sub_1685080(v64, v63);
      v68 = v84;
      v69 = v78;
      v70 = v65;
      if ( !v65 )
      {
        v80 = v84;
        v86 = v69;
        sub_1683C30(v64, v63, v66, 0, v68, v67, 0);
        v70 = v76;
        v68 = v80;
        v69 = v86;
      }
      v75 = v69;
      *(_QWORD *)(v68 + 8) = v63;
      *(_QWORD *)(v68 + 16) = v63;
      *(_QWORD *)(v68 + 24) = a1;
      *(_QWORD *)(v68 + 32) = v70;
      v79 = (_QWORD *)v70;
      *(_BYTE *)(v68 + 40) = 1;
      _InterlockedAdd(&dword_4F9F340, 1u);
      v71 = dword_4F9F340;
      *(_DWORD *)(v68 + 48) = v3;
      *(_DWORD *)(v68 + 44) = v71;
      v85 = v68;
      *(_QWORD *)v68 = sub_1684840(*(_QWORD *)(a1 + 2112), v3, v66, v70);
      sub_1684190(*(_QWORD *)(a1 + 2112), v3, v85);
      v72 = v79;
      v73 = (char *)v79 + v63;
      if ( v73 > (char *)v79 )
      {
        while ( 1 )
        {
          *v72 = v39;
          v39 = v72;
          v72[1] = v85;
          if ( v73 <= (char *)v72 + v3 )
            break;
          v72 = (_QWORD *)((char *)v72 + v3);
        }
      }
      else
      {
        v72 = 0;
      }
      *(_QWORD *)(a1 + 8 * v75 + 2128) = v72;
      sub_1684D30(*(_QWORD *)(v85 + 32) >> 3, *(_QWORD *)(v85 + 16) >> 3, v85);
      ++*(_DWORD *)(a1 + 44);
      sub_1683C60(v82);
      sub_1684B50(&qword_4F9F360);
      --dword_4F9F34C;
      j__pthread_mutex_unlock(qword_4F9F360);
      v39 = *(_QWORD **)(a1 + 8 * v38 + 2128);
    }
    *(_QWORD *)(a1 + 8 * v38 + 2128) = *v39;
    *(_QWORD *)(v39[1] + 8LL) -= v3;
LABEL_26:
    j__pthread_mutex_unlock(*(pthread_mutex_t **)(a1 + 7128));
    return v39;
  }
  v44 = a2;
  v39 = (_QWORD *)sub_1688B40(a2, 0);
  if ( !v39 )
  {
    if ( !dword_4F9F34C )
      goto LABEL_43;
    if ( qword_4F9F358 )
    {
      sub_1684B50(&qword_4F9F360);
      if ( qword_4F9F358 )
      {
        sub_1688C60(qword_4F9F358, 1);
        qword_4F9F358 = 0;
        dword_4F9F350 = 0;
      }
      j__pthread_mutex_unlock(qword_4F9F360);
    }
    v44 = a2;
    v49 = sub_1688B40(a2, 0);
    if ( v49 )
      return (_QWORD *)v49;
    else
LABEL_43:
      sub_1683C30(v44, 0, v45, v46, v47, v48, v74);
  }
  return v39;
}
