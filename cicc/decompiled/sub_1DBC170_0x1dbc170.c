// Function: sub_1DBC170
// Address: 0x1dbc170
//
void __fastcall sub_1DBC170(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  __int64 v6; // rcx
  __int64 v8; // r12
  unsigned int v9; // edx
  __int64 v10; // r13
  __int64 v11; // rax
  unsigned __int64 v12; // rax
  unsigned int v13; // r12d
  __int64 v14; // rdx
  __int64 v15; // rcx
  unsigned int v16; // edx
  __int16 v17; // ax
  _WORD *v18; // rcx
  __int16 *v19; // rdx
  unsigned __int16 v20; // r15
  __int64 v21; // rax
  __int64 *v22; // rax
  __int16 v23; // ax
  __int16 *v24; // r13
  __int64 v25; // rbx
  _QWORD *v26; // rax
  __int64 v27; // rdx
  _QWORD *v28; // rbx
  unsigned int v29; // r15d
  __int64 v30; // r13
  __int64 v31; // rax
  unsigned __int64 v32; // rax
  __int64 v33; // rdi
  unsigned __int64 v34; // rsi
  __int64 v35; // r9
  __int64 v36; // r10
  __int64 v37; // r11
  __int64 v38; // rax
  __int64 *v39; // rdx
  unsigned int v40; // ecx
  __int64 v41; // rax
  char v42; // r10
  __int64 v43; // rax
  __int64 v44; // rcx
  char v45; // si
  unsigned __int16 v46; // dx
  __int64 v47; // rax
  __int64 *i; // rbx
  __int64 **v49; // rax
  __int64 v50; // r13
  __int64 v51; // rax
  __int64 v52; // rsi
  _QWORD *v53; // rcx
  _QWORD *v54; // rax
  unsigned __int64 v55; // rsi
  unsigned int v56; // r15d
  _BYTE *v57; // r13
  __int64 v58; // r9
  __int64 v59; // r11
  __int64 v60; // rcx
  __int64 *v61; // rax
  unsigned int v62; // edx
  __int64 v63; // rcx
  int v65; // [rsp+34h] [rbp-12Ch]
  __int64 v66; // [rsp+38h] [rbp-128h]
  char v67; // [rsp+40h] [rbp-120h]
  __int64 v68; // [rsp+40h] [rbp-120h]
  unsigned int v69; // [rsp+40h] [rbp-120h]
  int v70; // [rsp+4Ch] [rbp-114h]
  _BYTE *v71; // [rsp+50h] [rbp-110h] BYREF
  __int64 v72; // [rsp+58h] [rbp-108h]
  _BYTE v73[64]; // [rsp+60h] [rbp-100h] BYREF
  _BYTE *v74; // [rsp+A0h] [rbp-C0h] BYREF
  __int64 v75; // [rsp+A8h] [rbp-B8h]
  _BYTE v76[176]; // [rsp+B0h] [rbp-B0h] BYREF

  v6 = *(_QWORD *)(a1 + 240);
  v74 = v76;
  v75 = 0x800000000LL;
  v71 = v73;
  v72 = 0x400000000LL;
  v65 = *(_DWORD *)(v6 + 32);
  if ( !v65 )
    return;
  v70 = 0;
  while ( 1 )
  {
    v9 = v70 & 0x7FFFFFFF;
    v10 = v70 & 0x7FFFFFFF;
    v11 = *(_QWORD *)(*(_QWORD *)(v6 + 24) + 16 * v10 + 8);
    if ( !v11 )
      goto LABEL_5;
    while ( (*(_BYTE *)(v11 + 4) & 8) != 0 )
    {
      v11 = *(_QWORD *)(v11 + 32);
      if ( !v11 )
        goto LABEL_5;
    }
    v12 = *(unsigned int *)(a1 + 408);
    if ( (unsigned int)v12 <= v9 || (v8 = *(_QWORD *)(*(_QWORD *)(a1 + 400) + 8 * v10)) == 0 )
    {
      v13 = v9 + 1;
      if ( (unsigned int)v12 < v9 + 1 )
      {
        if ( v13 >= v12 )
        {
          if ( v13 > v12 )
          {
            if ( v13 > (unsigned __int64)*(unsigned int *)(a1 + 412) )
            {
              sub_16CD150(a1 + 400, (const void *)(a1 + 416), v13, 8, a5, a6);
              v12 = *(unsigned int *)(a1 + 408);
            }
            v14 = *(_QWORD *)(a1 + 400);
            v52 = *(_QWORD *)(a1 + 416);
            v53 = (_QWORD *)(v14 + 8LL * v13);
            v54 = (_QWORD *)(v14 + 8 * v12);
            if ( v53 != v54 )
            {
              do
                *v54++ = v52;
              while ( v53 != v54 );
              v14 = *(_QWORD *)(a1 + 400);
            }
            *(_DWORD *)(a1 + 408) = v13;
LABEL_12:
            *(_QWORD *)(v14 + 8 * v10) = sub_1DBA290(v70 | 0x80000000);
            v8 = *(_QWORD *)(*(_QWORD *)(a1 + 400) + 8 * v10);
            sub_1DBB110((_QWORD *)a1, v8);
            a5 = *(unsigned int *)(v8 + 8);
            if ( !(_DWORD)a5 )
              goto LABEL_5;
            goto LABEL_13;
          }
        }
        else
        {
          *(_DWORD *)(a1 + 408) = v13;
        }
      }
      v14 = *(_QWORD *)(a1 + 400);
      goto LABEL_12;
    }
    a5 = *(unsigned int *)(v8 + 8);
    if ( !(_DWORD)a5 )
      goto LABEL_5;
LABEL_13:
    v15 = *(_QWORD *)(a1 + 248);
    LODWORD(v75) = 0;
    if ( !v15 )
      BUG();
    v16 = *(_DWORD *)(*(_QWORD *)(v15 + 8) + 24LL * *(unsigned int *)(*(_QWORD *)(a2 + 264) + 4 * v10) + 16);
    v17 = v16 & 0xF;
    v18 = (_WORD *)(*(_QWORD *)(v15 + 56) + 2LL * (v16 >> 4));
    v19 = v18 + 1;
    v20 = *v18 + *(_DWORD *)(*(_QWORD *)(a2 + 264) + 4 * v10) * v17;
LABEL_20:
    v24 = v19;
    while ( v24 )
    {
      v25 = *(_QWORD *)(*(_QWORD *)(a1 + 672) + 8LL * v20);
      if ( !v25 )
      {
        v67 = qword_4FC4440[20];
        v26 = (_QWORD *)sub_22077B0(104);
        v27 = v20;
        v25 = (__int64)v26;
        if ( v26 )
        {
          *v26 = v26 + 2;
          v26[1] = 0x200000000LL;
          v26[8] = v26 + 10;
          v26[9] = 0x200000000LL;
          if ( v67 )
          {
            v47 = sub_22077B0(48);
            v27 = v20;
            if ( v47 )
            {
              *(_DWORD *)(v47 + 8) = 0;
              *(_QWORD *)(v47 + 16) = 0;
              *(_QWORD *)(v47 + 24) = v47 + 8;
              *(_QWORD *)(v47 + 32) = v47 + 8;
              *(_QWORD *)(v47 + 40) = 0;
            }
            *(_QWORD *)(v25 + 96) = v47;
          }
          else
          {
            v26[12] = 0;
          }
        }
        *(_QWORD *)(*(_QWORD *)(a1 + 672) + 8 * v27) = v25;
        sub_1DBA8F0((_QWORD *)a1, v25, v20);
      }
      if ( *(_DWORD *)(v25 + 8) )
      {
        a5 = sub_1DB3C70((__int64 *)v25, *(_QWORD *)(*(_QWORD *)v8 + 8LL));
        v21 = (unsigned int)v75;
        if ( (unsigned int)v75 >= HIDWORD(v75) )
        {
          v68 = a5;
          sub_16CD150((__int64)&v74, v76, 0, 16, a5, a6);
          v21 = (unsigned int)v75;
          a5 = v68;
        }
        v22 = (__int64 *)&v74[16 * v21];
        *v22 = v25;
        v22[1] = a5;
        LODWORD(v75) = v75 + 1;
      }
      v23 = *v24;
      v19 = 0;
      ++v24;
      v20 += v23;
      if ( !v23 )
        goto LABEL_20;
    }
    if ( *(_BYTE *)(*(_QWORD *)(a1 + 240) + 16LL) )
    {
      LODWORD(v72) = 0;
      for ( i = *(__int64 **)(v8 + 104); i; LODWORD(v72) = v72 + 1 )
      {
        v50 = sub_1DB3C70(i, *(_QWORD *)(*(_QWORD *)v8 + 8LL));
        v51 = (unsigned int)v72;
        if ( (unsigned int)v72 >= HIDWORD(v72) )
        {
          sub_16CD150((__int64)&v71, v73, 0, 16, a5, a6);
          v51 = (unsigned int)v72;
        }
        v49 = (__int64 **)&v71[16 * v51];
        *v49 = i;
        v49[1] = (__int64 *)v50;
        i = (__int64 *)i[13];
      }
    }
    v28 = *(_QWORD **)v8;
    v29 = v70 | 0x80000000;
    v30 = *(_QWORD *)v8 + 24LL * *(unsigned int *)(v8 + 8);
    if ( v30 != *(_QWORD *)v8 )
    {
      while ( 1 )
      {
        v31 = v28[1];
        v28 += 3;
        if ( (v31 & 6) != 0 )
        {
          v32 = v31 & 0xFFFFFFFFFFFFFFF8LL;
          if ( v32 )
          {
            v33 = *(_QWORD *)(v32 + 16);
            if ( v33 )
              break;
          }
        }
LABEL_43:
        if ( (_QWORD *)v30 == v28 )
          goto LABEL_5;
      }
      v34 = (unsigned __int64)v74;
      v35 = (__int64)&v74[16 * (unsigned int)v75];
      if ( v74 != (_BYTE *)v35 )
      {
        do
        {
          a5 = *(_QWORD *)v34;
          v36 = *(_QWORD *)(v34 + 8);
          v37 = **(_QWORD **)v34;
          v38 = 24LL * *(unsigned int *)(*(_QWORD *)v34 + 8LL);
          v39 = (__int64 *)(v37 + v38);
          if ( v36 != v37 + v38 )
          {
            v40 = *(_DWORD *)((*(v28 - 2) & 0xFFFFFFFFFFFFFFF8LL) + 24) | ((__int64)*(v28 - 2) >> 1) & 3;
            if ( v40 < (*(_DWORD *)((*(_QWORD *)(v37 + v38 - 16) & 0xFFFFFFFFFFFFFFF8LL) + 24)
                      | (unsigned int)(*(__int64 *)(v37 + v38 - 16) >> 1) & 3) )
            {
              v39 = *(__int64 **)(v34 + 8);
              if ( v40 >= (*(_DWORD *)((*(_QWORD *)(v36 + 8) & 0xFFFFFFFFFFFFFFF8LL) + 24)
                         | (unsigned int)(*(__int64 *)(v36 + 8) >> 1) & 3) )
              {
                do
                {
                  v41 = v39[4];
                  v39 += 3;
                }
                while ( v40 >= (*(_DWORD *)((v41 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v41 >> 1) & 3) );
              }
            }
            *(_QWORD *)(v34 + 8) = v39;
            if ( v39 != (__int64 *)(*(_QWORD *)a5 + 24LL * *(unsigned int *)(a5 + 8))
              && (*(_DWORD *)((*v39 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v39 >> 1) & 3) < (*(_DWORD *)((*(v28 - 2) & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)((__int64)*(v28 - 2) >> 1) & 3) )
            {
              goto LABEL_42;
            }
          }
          v34 += 16LL;
        }
        while ( v35 != v34 );
      }
      v42 = *(_BYTE *)(*(_QWORD *)(a1 + 240) + 16LL);
      if ( !v42 )
        goto LABEL_84;
      v35 = 0xFFFFFFFFLL;
      if ( (_DWORD)v72 )
      {
        v55 = (unsigned __int64)v71;
        v69 = v29;
        v66 = v30;
        v56 = 0;
        v57 = &v71[16 * (unsigned int)v72];
        do
        {
          a5 = *(_QWORD *)v55;
          v58 = *(_QWORD *)(v55 + 8);
          v59 = **(_QWORD **)v55;
          v60 = 24LL * *(unsigned int *)(*(_QWORD *)v55 + 8LL);
          v61 = (__int64 *)(v59 + v60);
          if ( v58 != v59 + v60 )
          {
            v62 = *(_DWORD *)((*(v28 - 2) & 0xFFFFFFFFFFFFFFF8LL) + 24) | ((__int64)*(v28 - 2) >> 1) & 3;
            if ( v62 < (*(_DWORD *)((*(_QWORD *)(v59 + v60 - 16) & 0xFFFFFFFFFFFFFFF8LL) + 24)
                      | (unsigned int)(*(__int64 *)(v59 + v60 - 16) >> 1) & 3) )
            {
              v61 = *(__int64 **)(v55 + 8);
              if ( (*(_DWORD *)((*(_QWORD *)(v58 + 8) & 0xFFFFFFFFFFFFFFF8LL) + 24)
                  | (unsigned int)(*(__int64 *)(v58 + 8) >> 1) & 3) <= v62 )
              {
                do
                {
                  v63 = v61[4];
                  v61 += 3;
                }
                while ( v62 >= (*(_DWORD *)((v63 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v63 >> 1) & 3) );
              }
            }
            *(_QWORD *)(v55 + 8) = v61;
            if ( v61 != (__int64 *)(*(_QWORD *)a5 + 24LL * *(unsigned int *)(a5 + 8))
              && (*(_DWORD *)((*v61 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v61 >> 1) & 3) < (*(_DWORD *)((*(v28 - 2) & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)((__int64)*(v28 - 2) >> 1) & 3) )
            {
              v56 |= *(_DWORD *)(a5 + 112);
            }
          }
          v55 += 16LL;
        }
        while ( v57 != (_BYTE *)v55 );
        v35 = v56;
        v30 = v66;
        v29 = v69;
      }
      v43 = *(_QWORD *)(v33 + 32);
      v44 = v43 + 40LL * *(unsigned int *)(v33 + 40);
      if ( v43 == v44 )
        goto LABEL_82;
      v45 = 0;
      a5 = (unsigned int)~(_DWORD)v35;
      while ( 1 )
      {
LABEL_52:
        if ( *(_BYTE *)v43 || v29 != *(_DWORD *)(v43 + 8) )
          goto LABEL_51;
        v46 = (*(_DWORD *)v43 >> 8) & 0xFFF;
        if ( (*(_BYTE *)(v43 + 3) & 0x10) == 0 )
          break;
        if ( !v46 )
          v45 = v42;
        v43 += 40;
        if ( v44 == v43 )
        {
LABEL_58:
          if ( v45 )
            goto LABEL_84;
LABEL_82:
          if ( v28 == (_QWORD *)(*(_QWORD *)v8 + 24LL * *(unsigned int *)(v8 + 8)) || *v28 != *(v28 - 2) )
          {
LABEL_84:
            sub_1E1AFE0(v33, v29, 0, 0, a5, v35);
            goto LABEL_43;
          }
LABEL_42:
          sub_1E1A450(v33, v29, 0);
          goto LABEL_43;
        }
      }
      v35 = *(_QWORD *)(*(_QWORD *)(a1 + 248) + 248LL);
      if ( (*(_DWORD *)(v35 + 4LL * v46) & (unsigned int)a5) != 0 )
        goto LABEL_42;
LABEL_51:
      v43 += 40;
      if ( v44 == v43 )
        goto LABEL_58;
      goto LABEL_52;
    }
LABEL_5:
    if ( v65 == ++v70 )
      break;
    v6 = *(_QWORD *)(a1 + 240);
  }
  if ( v71 != v73 )
    _libc_free((unsigned __int64)v71);
  if ( v74 != v76 )
    _libc_free((unsigned __int64)v74);
}
