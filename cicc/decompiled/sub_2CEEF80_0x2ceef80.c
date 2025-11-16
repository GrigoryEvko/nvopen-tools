// Function: sub_2CEEF80
// Address: 0x2ceef80
//
void __fastcall sub_2CEEF80(_QWORD *a1, __int64 a2, __int64 a3)
{
  unsigned __int64 *v4; // rdx
  __int64 v5; // rax
  unsigned int v7; // eax
  unsigned __int64 v8; // rbx
  unsigned __int64 *v9; // r15
  unsigned __int64 v10; // rax
  __int64 v11; // r9
  _QWORD *v12; // rcx
  __int64 v13; // rax
  unsigned __int64 v14; // r8
  char *v15; // rbx
  unsigned __int8 v16; // dl
  __int64 v17; // r8
  int v18; // r11d
  char **v19; // r10
  unsigned int v20; // edi
  char **v21; // rcx
  char *v22; // rdx
  int v23; // edx
  __int64 v24; // r15
  __int64 v25; // r14
  __int64 v26; // rax
  unsigned __int64 v27; // rdx
  unsigned __int64 v28; // r12
  unsigned __int64 *v29; // r14
  unsigned __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // r15
  unsigned __int64 v33; // rbx
  unsigned __int64 *v34; // r15
  unsigned __int64 v35; // rax
  __int64 v36; // r15
  __int64 v37; // r15
  __int64 v38; // rax
  unsigned __int64 *v39; // r15
  unsigned __int64 v40; // rax
  __int64 v41; // r8
  __int64 v42; // r9
  __int64 v43; // rax
  __int64 v44; // r15
  unsigned __int64 v45; // rdx
  unsigned int v46; // eax
  char *v47; // rsi
  unsigned int v48; // r15d
  char **v49; // rdi
  char *v50; // rcx
  __int64 v51; // [rsp+10h] [rbp-A0h]
  unsigned __int64 v52; // [rsp+10h] [rbp-A0h]
  int v53; // [rsp+1Ch] [rbp-94h]
  __int64 v54; // [rsp+20h] [rbp-90h] BYREF
  __int64 v55; // [rsp+28h] [rbp-88h]
  __int64 v56; // [rsp+30h] [rbp-80h]
  __int64 v57; // [rsp+38h] [rbp-78h]
  _QWORD *v58; // [rsp+40h] [rbp-70h] BYREF
  __int64 v59; // [rsp+48h] [rbp-68h]
  _QWORD v60[12]; // [rsp+50h] [rbp-60h] BYREF

  v4 = (unsigned __int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v5 = *(_QWORD *)(a2 - 32);
  if ( !v5 || *(_BYTE *)v5 || *(_QWORD *)(v5 + 24) != *(_QWORD *)(a2 + 80) )
    BUG();
  v7 = *(_DWORD *)(v5 + 36);
  if ( v7 == 8930 )
  {
    v53 = 5;
    goto LABEL_9;
  }
  if ( v7 <= 0x22E2 )
  {
    if ( v7 == 8927 )
    {
      v53 = 4;
    }
    else
    {
      if ( v7 != 8928 )
        return;
      v53 = 1;
    }
LABEL_9:
    v8 = *v4;
    if ( *(_BYTE *)(*(_QWORD *)(*v4 + 8) + 8LL) != 14 )
      return;
    goto LABEL_13;
  }
  if ( v7 == 8931 )
  {
    v53 = 3;
    v8 = *v4;
    if ( *(_BYTE *)(*(_QWORD *)(*v4 + 8) + 8LL) == 14 )
    {
LABEL_13:
      v9 = (unsigned __int64 *)(*a1 + 120LL);
      v10 = sub_22077B0(0x20u);
      if ( v10 )
        *(_QWORD *)v10 = 0;
      *(_QWORD *)(v10 + 8) = v8;
      *(_QWORD *)(v10 + 16) = a3;
      *(_DWORD *)(v10 + 24) = v53;
      sub_2CE79A0(v9, 0, (_QWORD *)(v10 + 8), v8, v10);
      v12 = v60;
      v54 = 0;
      v58 = v60;
      v55 = 0;
      v56 = 0;
      v57 = 0;
      v60[0] = v8;
      v59 = 0x600000001LL;
      LODWORD(v13) = 1;
      while ( 1 )
      {
        v14 = (unsigned int)v13;
        v13 = (unsigned int)(v13 - 1);
        v15 = (char *)v12[v14 - 1];
        LODWORD(v59) = v13;
        v16 = *v15;
        if ( (unsigned __int8)*v15 <= 0x1Cu )
          goto LABEL_22;
        switch ( v16 )
        {
          case 'N':
            v36 = *((_QWORD *)v15 - 4);
            if ( v14 > HIDWORD(v59) )
            {
              sub_C8D5F0((__int64)&v58, v60, v14, 8u, v14, v11);
              v12 = v58;
              v13 = (unsigned int)v59;
            }
            v12[v13] = v36;
            LODWORD(v59) = v59 + 1;
            v33 = *((_QWORD *)v15 - 4);
            goto LABEL_51;
          case '?':
            goto LABEL_48;
          case 'V':
            v37 = *((_QWORD *)v15 - 12);
            if ( v14 > HIDWORD(v59) )
            {
              sub_C8D5F0((__int64)&v58, v60, v14, 8u, v14, v11);
              v12 = v58;
              v13 = (unsigned int)v59;
            }
            v12[v13] = v37;
            v38 = *a1;
            LODWORD(v59) = v59 + 1;
            v39 = (unsigned __int64 *)(v38 + 120);
            v52 = *((_QWORD *)v15 - 12);
            v40 = sub_22077B0(0x20u);
            if ( v40 )
              *(_QWORD *)v40 = 0;
            *(_QWORD *)(v40 + 8) = v52;
            *(_QWORD *)(v40 + 16) = a3;
            *(_DWORD *)(v40 + 24) = v53;
            sub_2CE79A0(v39, 0, (_QWORD *)(v40 + 8), v52, v40);
            v43 = (unsigned int)v59;
            v44 = *((_QWORD *)v15 - 8);
            v45 = (unsigned int)v59 + 1LL;
            if ( v45 > HIDWORD(v59) )
            {
              sub_C8D5F0((__int64)&v58, v60, v45, 8u, v41, v42);
              v43 = (unsigned int)v59;
            }
            v58[v43] = v44;
            LODWORD(v59) = v59 + 1;
            v33 = *((_QWORD *)v15 - 8);
            goto LABEL_51;
        }
        if ( v16 != 84 )
        {
          if ( v16 == 85 )
          {
            v31 = *((_QWORD *)v15 - 4);
            if ( v31 )
            {
              if ( !*(_BYTE *)v31
                && *(_QWORD *)(v31 + 24) == *((_QWORD *)v15 + 10)
                && (*(_BYTE *)(v31 + 33) & 0x20) != 0
                && *(_DWORD *)(v31 + 36) == 8170 )
              {
LABEL_48:
                v32 = *(_QWORD *)&v15[-32 * (*((_DWORD *)v15 + 1) & 0x7FFFFFF)];
                if ( v14 > HIDWORD(v59) )
                {
                  sub_C8D5F0((__int64)&v58, v60, v14, 8u, v14, v11);
                  v12 = v58;
                  v13 = (unsigned int)v59;
                }
                v12[v13] = v32;
                LODWORD(v59) = v59 + 1;
                v33 = *(_QWORD *)&v15[-32 * (*((_DWORD *)v15 + 1) & 0x7FFFFFF)];
LABEL_51:
                v34 = (unsigned __int64 *)(*a1 + 120LL);
                v35 = sub_22077B0(0x20u);
                if ( v35 )
                  *(_QWORD *)v35 = 0;
                *(_QWORD *)(v35 + 8) = v33;
                *(_QWORD *)(v35 + 16) = a3;
                *(_DWORD *)(v35 + 24) = v53;
                sub_2CE79A0(v34, 0, (_QWORD *)(v35 + 8), v33, v35);
LABEL_54:
                LODWORD(v13) = v59;
              }
            }
          }
LABEL_22:
          if ( !(_DWORD)v13 )
            goto LABEL_41;
          goto LABEL_23;
        }
        if ( !(_DWORD)v57 )
          break;
        v11 = (unsigned int)(v57 - 1);
        v17 = v55;
        v18 = 1;
        v19 = 0;
        v20 = v11 & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
        v21 = (char **)(v55 + 8LL * v20);
        v22 = *v21;
        if ( v15 == *v21 )
          goto LABEL_22;
        while ( v22 != (char *)-4096LL )
        {
          if ( v19 || v22 != (char *)-8192LL )
            v21 = v19;
          v20 = v11 & (v18 + v20);
          v22 = *(char **)(v55 + 8LL * v20);
          if ( v15 == v22 )
            goto LABEL_22;
          ++v18;
          v19 = v21;
          v21 = (char **)(v55 + 8LL * v20);
        }
        if ( !v19 )
          v19 = v21;
        ++v54;
        v23 = v56 + 1;
        if ( 4 * ((int)v56 + 1) >= (unsigned int)(3 * v57) )
          goto LABEL_68;
        if ( (int)v57 - HIDWORD(v56) - v23 <= (unsigned int)v57 >> 3 )
        {
          sub_110B120((__int64)&v54, v57);
          if ( !(_DWORD)v57 )
          {
LABEL_96:
            LODWORD(v56) = v56 + 1;
            BUG();
          }
          v17 = 1;
          v48 = (v57 - 1) & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
          v19 = (char **)(v55 + 8LL * v48);
          v23 = v56 + 1;
          v49 = 0;
          v50 = *v19;
          if ( v15 != *v19 )
          {
            while ( v50 != (char *)-4096LL )
            {
              if ( v50 == (char *)-8192LL && !v49 )
                v49 = v19;
              v11 = (unsigned int)(v17 + 1);
              v48 = (v57 - 1) & (v17 + v48);
              v19 = (char **)(v55 + 8LL * v48);
              v50 = *v19;
              if ( v15 == *v19 )
                goto LABEL_31;
              v17 = (unsigned int)v11;
            }
            if ( v49 )
              v19 = v49;
          }
        }
LABEL_31:
        LODWORD(v56) = v23;
        if ( *v19 != (char *)-4096LL )
          --HIDWORD(v56);
        *v19 = v15;
        v24 = 0;
        if ( (*((_DWORD *)v15 + 1) & 0x7FFFFFF) == 0 )
          goto LABEL_54;
        v51 = a3;
        do
        {
          v25 = *(_QWORD *)(*((_QWORD *)v15 - 1) + 32 * v24);
          v26 = (unsigned int)v59;
          v27 = (unsigned int)v59 + 1LL;
          if ( v27 > HIDWORD(v59) )
          {
            sub_C8D5F0((__int64)&v58, v60, v27, 8u, v17, v11);
            v26 = (unsigned int)v59;
          }
          v58[v26] = v25;
          LODWORD(v59) = v59 + 1;
          v28 = *(_QWORD *)(*((_QWORD *)v15 - 1) + 32 * v24);
          v29 = (unsigned __int64 *)(*a1 + 120LL);
          v30 = sub_22077B0(0x20u);
          if ( v30 )
            *(_QWORD *)v30 = 0;
          *(_QWORD *)(v30 + 8) = v28;
          ++v24;
          *(_QWORD *)(v30 + 16) = v51;
          *(_DWORD *)(v30 + 24) = v53;
          sub_2CE79A0(v29, 0, (_QWORD *)(v30 + 8), v28, v30);
        }
        while ( (*((_DWORD *)v15 + 1) & 0x7FFFFFFu) > (unsigned int)v24 );
        LODWORD(v13) = v59;
        a3 = v51;
        if ( !(_DWORD)v59 )
        {
LABEL_41:
          sub_C7D6A0(v55, 8LL * (unsigned int)v57, 8);
          if ( v58 != v60 )
            _libc_free((unsigned __int64)v58);
          return;
        }
LABEL_23:
        v12 = v58;
      }
      ++v54;
LABEL_68:
      sub_110B120((__int64)&v54, 2 * v57);
      if ( !(_DWORD)v57 )
        goto LABEL_96;
      v46 = (v57 - 1) & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
      v19 = (char **)(v55 + 8LL * v46);
      v47 = *v19;
      v23 = v56 + 1;
      if ( v15 != *v19 )
      {
        v11 = 1;
        v17 = 0;
        while ( v47 != (char *)-4096LL )
        {
          if ( v47 == (char *)-8192LL && !v17 )
            v17 = (__int64)v19;
          v46 = (v57 - 1) & (v11 + v46);
          v19 = (char **)(v55 + 8LL * v46);
          v47 = *v19;
          if ( v15 == *v19 )
            goto LABEL_31;
          v11 = (unsigned int)(v11 + 1);
        }
        if ( v17 )
          v19 = (char **)v17;
      }
      goto LABEL_31;
    }
  }
}
