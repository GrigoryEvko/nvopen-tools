// Function: sub_3541720
// Address: 0x3541720
//
__int64 __fastcall sub_3541720(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  __int64 v7; // rcx
  __int64 v8; // rcx
  __int64 v9; // rbx
  __int64 v10; // r15
  __int64 v11; // r12
  __int64 v12; // rsi
  __int64 v13; // rdx
  int v14; // edx
  char v15; // dl
  _BYTE *v16; // rdi
  __int64 v17; // rdx
  __int64 v18; // rdi
  __int64 v19; // rsi
  __int64 v20; // r8
  __int64 v21; // rdx
  unsigned __int64 v22; // rdi
  _BYTE *v23; // rsi
  char v24; // dl
  __int64 v25; // rcx
  __int64 v26; // rbx
  __int64 v27; // r15
  __int64 v28; // r12
  __int64 v29; // rsi
  __int64 v30; // rsi
  __int64 v31; // rdx
  int v32; // edx
  char v33; // dl
  _BYTE *v34; // rdi
  int v35; // edx
  __int64 v36; // rdi
  __int64 v37; // rdi
  __int64 v38; // rdx
  unsigned __int64 v39; // rdi
  _BYTE *v40; // rsi
  char v41; // dl
  __int64 v43; // r12
  __int64 v44; // r15
  __int64 v46; // rcx
  __int64 v47; // rdx
  int v48; // edx
  char v49; // dl
  _BYTE *v50; // rdi
  __int64 v51; // rdx
  int v52; // r10d
  __int64 v53; // rdi
  __int64 v54; // rsi
  __int64 v55; // rdx
  unsigned __int64 v56; // rdi
  _BYTE *v57; // rcx
  char v58; // dl
  unsigned __int64 v59; // rdi
  __int64 v60; // [rsp+8h] [rbp-C8h]
  __int64 v61; // [rsp+18h] [rbp-B8h]
  __int64 v62; // [rsp+20h] [rbp-B0h]
  __int64 v63; // [rsp+28h] [rbp-A8h]
  __int64 v64; // [rsp+30h] [rbp-A0h]
  __int64 v65; // [rsp+48h] [rbp-88h]
  __int64 v66; // [rsp+48h] [rbp-88h]
  __int64 v67; // [rsp+48h] [rbp-88h]
  int v68; // [rsp+50h] [rbp-80h]
  int v69; // [rsp+50h] [rbp-80h]
  int v70; // [rsp+50h] [rbp-80h]
  int v71; // [rsp+54h] [rbp-7Ch]
  int v72; // [rsp+54h] [rbp-7Ch]
  int v73; // [rsp+54h] [rbp-7Ch]
  int v74; // [rsp+58h] [rbp-78h]
  int v75; // [rsp+58h] [rbp-78h]
  int v76; // [rsp+58h] [rbp-78h]
  _BYTE *v77; // [rsp+60h] [rbp-70h] BYREF
  __int64 v78; // [rsp+68h] [rbp-68h]
  _BYTE v79[4]; // [rsp+70h] [rbp-60h] BYREF
  int v80; // [rsp+74h] [rbp-5Ch]
  int v81; // [rsp+78h] [rbp-58h]
  int v82; // [rsp+7Ch] [rbp-54h]
  int v83; // [rsp+80h] [rbp-50h]
  __int64 v84; // [rsp+88h] [rbp-48h]
  int v85; // [rsp+90h] [rbp-40h]

  v61 = a1;
  if ( a1 == a2 )
    return a3;
  if ( a2 == a3 )
    return a1;
  v60 = a1 + a3 - a2;
  v6 = 0x2E8BA2E8BA2E8BA3LL * ((a2 - a1) >> 3);
  v62 = 0x2E8BA2E8BA2E8BA3LL * ((a3 - a1) >> 3);
  v7 = v62 - v6;
  v63 = v6;
  if ( v6 != v62 - v6 )
  {
    while ( 1 )
    {
      v64 = v62 - v63;
      if ( v63 >= v62 - v63 )
      {
        v25 = v61 + 88 * v62;
        v61 = v25 - 88 * v64;
        if ( v63 > 0 )
        {
          v26 = v25 - 88 * v64 - 56;
          v27 = v25 - 40;
          v28 = 0;
          do
          {
            v35 = *(_DWORD *)(v26 - 16);
            v36 = *(_QWORD *)(v26 - 24);
            *(_DWORD *)(v26 - 16) = 0;
            ++*(_QWORD *)(v26 - 32);
            v69 = v35;
            v66 = v36;
            v37 = 0;
            v72 = *(_DWORD *)(v26 - 12);
            v75 = *(_DWORD *)(v26 - 8);
            v38 = *(unsigned int *)(v26 + 8);
            *(_QWORD *)(v26 - 24) = 0;
            *(_DWORD *)(v26 - 12) = 0;
            *(_DWORD *)(v26 - 8) = 0;
            v77 = v79;
            v78 = 0;
            if ( (_DWORD)v38 )
            {
              sub_353DE10((__int64)&v77, (char **)v26, v38, v25, a5, a6);
              v37 = *(_QWORD *)(v26 - 24);
            }
            v29 = *(unsigned int *)(v26 - 8);
            v79[0] = *(_BYTE *)(v26 + 16);
            v80 = *(_DWORD *)(v26 + 20);
            v81 = *(_DWORD *)(v26 + 24);
            v82 = *(_DWORD *)(v26 + 28);
            v83 = *(_DWORD *)(v26 + 32);
            v84 = *(_QWORD *)(v26 + 40);
            v85 = *(_DWORD *)(v26 + 48);
            sub_C7D6A0(v37, 8 * v29, 8);
            *(_DWORD *)(v26 - 8) = 0;
            *(_QWORD *)(v26 - 24) = 0;
            *(_DWORD *)(v26 - 16) = 0;
            *(_DWORD *)(v26 - 12) = 0;
            ++*(_QWORD *)(v26 - 32);
            v30 = *(_QWORD *)(v27 - 40);
            ++*(_QWORD *)(v27 - 48);
            v31 = *(_QWORD *)(v26 - 24);
            *(_QWORD *)(v26 - 24) = v30;
            LODWORD(v30) = *(_DWORD *)(v27 - 32);
            *(_QWORD *)(v27 - 40) = v31;
            LODWORD(v31) = *(_DWORD *)(v26 - 16);
            *(_DWORD *)(v26 - 16) = v30;
            LODWORD(v30) = *(_DWORD *)(v27 - 28);
            *(_DWORD *)(v27 - 32) = v31;
            LODWORD(v31) = *(_DWORD *)(v26 - 12);
            *(_DWORD *)(v26 - 12) = v30;
            LODWORD(v30) = *(_DWORD *)(v27 - 24);
            *(_DWORD *)(v27 - 28) = v31;
            LODWORD(v31) = *(_DWORD *)(v26 - 8);
            *(_DWORD *)(v26 - 8) = v30;
            *(_DWORD *)(v27 - 24) = v31;
            if ( v26 != v27 - 16 )
            {
              if ( *(_DWORD *)(v27 - 8) )
              {
                if ( *(_QWORD *)v26 != v26 + 16 )
                  _libc_free(*(_QWORD *)v26);
                *(_QWORD *)v26 = *(_QWORD *)(v27 - 16);
                *(_DWORD *)(v26 + 8) = *(_DWORD *)(v27 - 8);
                *(_DWORD *)(v26 + 12) = *(_DWORD *)(v27 - 4);
                *(_QWORD *)(v27 - 16) = v27;
                *(_DWORD *)(v27 - 4) = 0;
                *(_DWORD *)(v27 - 8) = 0;
              }
              else
              {
                *(_DWORD *)(v26 + 8) = 0;
              }
            }
            *(_BYTE *)(v26 + 16) = *(_BYTE *)v27;
            *(_DWORD *)(v26 + 20) = *(_DWORD *)(v27 + 4);
            *(_DWORD *)(v26 + 24) = *(_DWORD *)(v27 + 8);
            *(_DWORD *)(v26 + 28) = *(_DWORD *)(v27 + 12);
            *(_DWORD *)(v26 + 32) = *(_DWORD *)(v27 + 16);
            *(_QWORD *)(v26 + 40) = *(_QWORD *)(v27 + 24);
            *(_DWORD *)(v26 + 48) = *(_DWORD *)(v27 + 32);
            sub_C7D6A0(*(_QWORD *)(v27 - 40), 8LL * *(unsigned int *)(v27 - 24), 8);
            ++*(_QWORD *)(v27 - 48);
            *(_QWORD *)(v27 - 40) = v66;
            *(_DWORD *)(v27 - 32) = v69;
            *(_DWORD *)(v27 - 28) = v72;
            *(_DWORD *)(v27 - 24) = v75;
            v32 = v78;
            if ( (_DWORD)v78 )
            {
              v39 = *(_QWORD *)(v27 - 16);
              if ( v39 != v27 )
              {
                _libc_free(v39);
                v32 = v78;
              }
              *(_DWORD *)(v27 - 8) = v32;
              v40 = v77;
              *(_DWORD *)(v27 - 4) = HIDWORD(v78);
              v41 = v79[0];
              *(_QWORD *)(v27 - 16) = v40;
              *(_BYTE *)v27 = v41;
              *(_DWORD *)(v27 + 4) = v80;
              *(_DWORD *)(v27 + 8) = v81;
              *(_DWORD *)(v27 + 12) = v82;
              *(_DWORD *)(v27 + 16) = v83;
              *(_QWORD *)(v27 + 24) = v84;
              *(_DWORD *)(v27 + 32) = v85;
            }
            else
            {
              v33 = v79[0];
              v34 = v77;
              *(_DWORD *)(v27 - 8) = 0;
              *(_BYTE *)v27 = v33;
              *(_DWORD *)(v27 + 4) = v80;
              *(_DWORD *)(v27 + 8) = v81;
              *(_DWORD *)(v27 + 12) = v82;
              *(_DWORD *)(v27 + 16) = v83;
              *(_QWORD *)(v27 + 24) = v84;
              *(_DWORD *)(v27 + 32) = v85;
              if ( v34 != v79 )
                _libc_free((unsigned __int64)v34);
            }
            ++v28;
            v27 -= 88;
            v26 -= 88;
            sub_C7D6A0(0, 0, 8);
          }
          while ( v63 != v28 );
          v61 += -88 * v63;
        }
        v63 = v62 % v64;
        if ( !(v62 % v64) )
          return v60;
      }
      else
      {
        if ( v62 - v63 > 0 )
        {
          v8 = v61;
          v9 = v61 + 32;
          v10 = v61 + 88 * v63 + 48;
          v11 = 0;
          do
          {
            v17 = *(_QWORD *)(v9 - 24);
            v18 = 0;
            v19 = 0;
            v20 = *(unsigned int *)(v9 + 8);
            ++*(_QWORD *)(v9 - 32);
            v65 = v17;
            v68 = *(_DWORD *)(v9 - 16);
            LODWORD(v17) = *(_DWORD *)(v9 - 12);
            *(_QWORD *)(v9 - 24) = 0;
            v71 = v17;
            v21 = *(unsigned int *)(v9 - 8);
            *(_DWORD *)(v9 - 16) = 0;
            *(_DWORD *)(v9 - 12) = 0;
            v74 = v21;
            *(_DWORD *)(v9 - 8) = 0;
            v77 = v79;
            v78 = 0;
            if ( (_DWORD)v20 )
            {
              sub_353DE10((__int64)&v77, (char **)v9, v21, v8, v20, a6);
              v18 = *(_QWORD *)(v9 - 24);
              v19 = 8LL * *(unsigned int *)(v9 - 8);
            }
            v79[0] = *(_BYTE *)(v9 + 16);
            v80 = *(_DWORD *)(v9 + 20);
            v81 = *(_DWORD *)(v9 + 24);
            v82 = *(_DWORD *)(v9 + 28);
            v83 = *(_DWORD *)(v9 + 32);
            v84 = *(_QWORD *)(v9 + 40);
            v85 = *(_DWORD *)(v9 + 48);
            sub_C7D6A0(v18, v19, 8);
            *(_DWORD *)(v9 - 8) = 0;
            *(_QWORD *)(v9 - 24) = 0;
            *(_DWORD *)(v9 - 16) = 0;
            *(_DWORD *)(v9 - 12) = 0;
            ++*(_QWORD *)(v9 - 32);
            v12 = *(_QWORD *)(v10 - 40);
            ++*(_QWORD *)(v10 - 48);
            v13 = *(_QWORD *)(v9 - 24);
            *(_QWORD *)(v9 - 24) = v12;
            LODWORD(v12) = *(_DWORD *)(v10 - 32);
            *(_QWORD *)(v10 - 40) = v13;
            LODWORD(v13) = *(_DWORD *)(v9 - 16);
            *(_DWORD *)(v9 - 16) = v12;
            LODWORD(v12) = *(_DWORD *)(v10 - 28);
            *(_DWORD *)(v10 - 32) = v13;
            LODWORD(v13) = *(_DWORD *)(v9 - 12);
            *(_DWORD *)(v9 - 12) = v12;
            LODWORD(v12) = *(_DWORD *)(v10 - 24);
            *(_DWORD *)(v10 - 28) = v13;
            LODWORD(v13) = *(_DWORD *)(v9 - 8);
            *(_DWORD *)(v9 - 8) = v12;
            *(_DWORD *)(v10 - 24) = v13;
            if ( v9 != v10 - 16 )
            {
              if ( *(_DWORD *)(v10 - 8) )
              {
                if ( *(_QWORD *)v9 != v9 + 16 )
                  _libc_free(*(_QWORD *)v9);
                *(_QWORD *)v9 = *(_QWORD *)(v10 - 16);
                *(_DWORD *)(v9 + 8) = *(_DWORD *)(v10 - 8);
                *(_DWORD *)(v9 + 12) = *(_DWORD *)(v10 - 4);
                *(_QWORD *)(v10 - 16) = v10;
                *(_DWORD *)(v10 - 4) = 0;
                *(_DWORD *)(v10 - 8) = 0;
              }
              else
              {
                *(_DWORD *)(v9 + 8) = 0;
              }
            }
            *(_BYTE *)(v9 + 16) = *(_BYTE *)v10;
            *(_DWORD *)(v9 + 20) = *(_DWORD *)(v10 + 4);
            *(_DWORD *)(v9 + 24) = *(_DWORD *)(v10 + 8);
            *(_DWORD *)(v9 + 28) = *(_DWORD *)(v10 + 12);
            *(_DWORD *)(v9 + 32) = *(_DWORD *)(v10 + 16);
            *(_QWORD *)(v9 + 40) = *(_QWORD *)(v10 + 24);
            *(_DWORD *)(v9 + 48) = *(_DWORD *)(v10 + 32);
            sub_C7D6A0(*(_QWORD *)(v10 - 40), 8LL * *(unsigned int *)(v10 - 24), 8);
            ++*(_QWORD *)(v10 - 48);
            *(_QWORD *)(v10 - 40) = v65;
            *(_DWORD *)(v10 - 32) = v68;
            *(_DWORD *)(v10 - 28) = v71;
            *(_DWORD *)(v10 - 24) = v74;
            v14 = v78;
            if ( (_DWORD)v78 )
            {
              v22 = *(_QWORD *)(v10 - 16);
              if ( v22 != v10 )
              {
                _libc_free(v22);
                v14 = v78;
              }
              *(_DWORD *)(v10 - 8) = v14;
              v23 = v77;
              *(_DWORD *)(v10 - 4) = HIDWORD(v78);
              v24 = v79[0];
              *(_QWORD *)(v10 - 16) = v23;
              *(_BYTE *)v10 = v24;
              *(_DWORD *)(v10 + 4) = v80;
              *(_DWORD *)(v10 + 8) = v81;
              *(_DWORD *)(v10 + 12) = v82;
              *(_DWORD *)(v10 + 16) = v83;
              *(_QWORD *)(v10 + 24) = v84;
              *(_DWORD *)(v10 + 32) = v85;
            }
            else
            {
              v15 = v79[0];
              v16 = v77;
              *(_DWORD *)(v10 - 8) = 0;
              *(_BYTE *)v10 = v15;
              *(_DWORD *)(v10 + 4) = v80;
              *(_DWORD *)(v10 + 8) = v81;
              *(_DWORD *)(v10 + 12) = v82;
              *(_DWORD *)(v10 + 16) = v83;
              *(_QWORD *)(v10 + 24) = v84;
              *(_DWORD *)(v10 + 32) = v85;
              if ( v16 != v79 )
                _libc_free((unsigned __int64)v16);
            }
            ++v11;
            v10 += 88;
            v9 += 88;
            sub_C7D6A0(0, 0, 8);
          }
          while ( v64 != v11 );
          v61 += 88 * v64;
        }
        if ( !(v62 % v63) )
          return v60;
        v64 = v63;
        v63 -= v62 % v63;
      }
      v62 = v64;
    }
  }
  v43 = a1;
  v44 = a2 + 48;
  do
  {
    v51 = *(_QWORD *)(v43 + 8);
    v52 = *(_DWORD *)(v43 + 40);
    v77 = v79;
    v53 = 0;
    ++*(_QWORD *)v43;
    v54 = 0;
    v67 = v51;
    v70 = *(_DWORD *)(v43 + 16);
    LODWORD(v51) = *(_DWORD *)(v43 + 20);
    *(_QWORD *)(v43 + 8) = 0;
    v73 = v51;
    v55 = *(unsigned int *)(v43 + 24);
    *(_DWORD *)(v43 + 16) = 0;
    *(_DWORD *)(v43 + 20) = 0;
    v76 = v55;
    *(_DWORD *)(v43 + 24) = 0;
    v78 = 0;
    if ( v52 )
    {
      sub_353DE10((__int64)&v77, (char **)(v43 + 32), v55, v7, a5, a6);
      v53 = *(_QWORD *)(v43 + 8);
      v54 = 8LL * *(unsigned int *)(v43 + 24);
    }
    v79[0] = *(_BYTE *)(v43 + 48);
    v80 = *(_DWORD *)(v43 + 52);
    v81 = *(_DWORD *)(v43 + 56);
    v82 = *(_DWORD *)(v43 + 60);
    v83 = *(_DWORD *)(v43 + 64);
    v84 = *(_QWORD *)(v43 + 72);
    v85 = *(_DWORD *)(v43 + 80);
    sub_C7D6A0(v53, v54, 8);
    ++*(_QWORD *)v43;
    *(_DWORD *)(v43 + 24) = 0;
    *(_QWORD *)(v43 + 8) = 0;
    *(_DWORD *)(v43 + 16) = 0;
    *(_DWORD *)(v43 + 20) = 0;
    v46 = *(_QWORD *)(v44 - 40);
    ++*(_QWORD *)(v44 - 48);
    v47 = *(_QWORD *)(v43 + 8);
    *(_QWORD *)(v43 + 8) = v46;
    LODWORD(v46) = *(_DWORD *)(v44 - 32);
    *(_QWORD *)(v44 - 40) = v47;
    LODWORD(v47) = *(_DWORD *)(v43 + 16);
    *(_DWORD *)(v43 + 16) = v46;
    LODWORD(v46) = *(_DWORD *)(v44 - 28);
    *(_DWORD *)(v44 - 32) = v47;
    LODWORD(v47) = *(_DWORD *)(v43 + 20);
    *(_DWORD *)(v43 + 20) = v46;
    LODWORD(v46) = *(_DWORD *)(v44 - 24);
    *(_DWORD *)(v44 - 28) = v47;
    LODWORD(v47) = *(_DWORD *)(v43 + 24);
    *(_DWORD *)(v43 + 24) = v46;
    *(_DWORD *)(v44 - 24) = v47;
    if ( v43 + 32 != v44 - 16 )
    {
      if ( *(_DWORD *)(v44 - 8) )
      {
        v59 = *(_QWORD *)(v43 + 32);
        if ( v59 != v43 + 48 )
          _libc_free(v59);
        *(_QWORD *)(v43 + 32) = *(_QWORD *)(v44 - 16);
        *(_DWORD *)(v43 + 40) = *(_DWORD *)(v44 - 8);
        *(_DWORD *)(v43 + 44) = *(_DWORD *)(v44 - 4);
        *(_QWORD *)(v44 - 16) = v44;
        *(_DWORD *)(v44 - 4) = 0;
        *(_DWORD *)(v44 - 8) = 0;
      }
      else
      {
        *(_DWORD *)(v43 + 40) = 0;
      }
    }
    *(_BYTE *)(v43 + 48) = *(_BYTE *)v44;
    *(_DWORD *)(v43 + 52) = *(_DWORD *)(v44 + 4);
    *(_DWORD *)(v43 + 56) = *(_DWORD *)(v44 + 8);
    *(_DWORD *)(v43 + 60) = *(_DWORD *)(v44 + 12);
    *(_DWORD *)(v43 + 64) = *(_DWORD *)(v44 + 16);
    *(_QWORD *)(v43 + 72) = *(_QWORD *)(v44 + 24);
    *(_DWORD *)(v43 + 80) = *(_DWORD *)(v44 + 32);
    sub_C7D6A0(*(_QWORD *)(v44 - 40), 8LL * *(unsigned int *)(v44 - 24), 8);
    ++*(_QWORD *)(v44 - 48);
    *(_QWORD *)(v44 - 40) = v67;
    *(_DWORD *)(v44 - 32) = v70;
    *(_DWORD *)(v44 - 28) = v73;
    *(_DWORD *)(v44 - 24) = v76;
    v48 = v78;
    if ( (_DWORD)v78 )
    {
      v56 = *(_QWORD *)(v44 - 16);
      if ( v56 != v44 )
      {
        _libc_free(v56);
        v48 = v78;
      }
      *(_DWORD *)(v44 - 8) = v48;
      v57 = v77;
      *(_DWORD *)(v44 - 4) = HIDWORD(v78);
      v58 = v79[0];
      *(_QWORD *)(v44 - 16) = v57;
      *(_BYTE *)v44 = v58;
      *(_DWORD *)(v44 + 4) = v80;
      *(_DWORD *)(v44 + 8) = v81;
      *(_DWORD *)(v44 + 12) = v82;
      *(_DWORD *)(v44 + 16) = v83;
      *(_QWORD *)(v44 + 24) = v84;
      *(_DWORD *)(v44 + 32) = v85;
    }
    else
    {
      v49 = v79[0];
      v50 = v77;
      *(_DWORD *)(v44 - 8) = 0;
      *(_BYTE *)v44 = v49;
      *(_DWORD *)(v44 + 4) = v80;
      *(_DWORD *)(v44 + 8) = v81;
      *(_DWORD *)(v44 + 12) = v82;
      *(_DWORD *)(v44 + 16) = v83;
      *(_QWORD *)(v44 + 24) = v84;
      *(_DWORD *)(v44 + 32) = v85;
      if ( v50 != v79 )
        _libc_free((unsigned __int64)v50);
    }
    v43 += 88;
    v44 += 88;
    sub_C7D6A0(0, 0, 8);
  }
  while ( a2 != v43 );
  return v43;
}
