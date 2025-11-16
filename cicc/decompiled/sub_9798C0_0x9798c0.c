// Function: sub_9798C0
// Address: 0x9798c0
//
__int64 __fastcall sub_9798C0(
        char *a1,
        unsigned __int64 a2,
        unsigned int a3,
        __int64 a4,
        __int64 *a5,
        unsigned __int64 a6,
        _BYTE *a7,
        __int64 *a8,
        __int64 a9)
{
  _QWORD *v9; // rax
  unsigned __int64 v13; // r12
  _QWORD *i; // rdx
  __int64 *v15; // rax
  __int64 v16; // rsi
  __int64 *j; // rdx
  __int64 v18; // rax
  __int64 v19; // r14
  __int64 v20; // rdx
  _BYTE *v21; // rsi
  _QWORD *v22; // r15
  __int64 v23; // rbx
  _QWORD *v24; // r13
  _QWORD *v25; // rbx
  __int64 v26; // rax
  _BYTE *v27; // rax
  __int64 v29; // r15
  __int64 v30; // r12
  __int64 v31; // rax
  __int64 v32; // rax
  unsigned int m; // ebx
  __int64 v34; // r13
  __int64 v35; // rax
  unsigned __int64 v36; // rdx
  _BYTE *v37; // r15
  __int64 v38; // r12
  int v39; // eax
  __int64 v40; // rax
  unsigned __int64 v41; // rdx
  __int64 v42; // rax
  unsigned __int64 v43; // rdx
  __int64 v44; // rax
  unsigned __int64 v45; // rdx
  _BYTE *v46; // rsi
  _QWORD *v47; // r13
  __int64 v48; // rbx
  unsigned __int64 k; // r15
  __int64 v50; // rax
  _BYTE *v51; // rdi
  __int64 v52; // [rsp+8h] [rbp-1E8h]
  _QWORD *v55; // [rsp+38h] [rbp-1B8h]
  int v56; // [rsp+38h] [rbp-1B8h]
  __int64 v57; // [rsp+40h] [rbp-1B0h]
  __int64 v58; // [rsp+40h] [rbp-1B0h]
  __int64 v59; // [rsp+48h] [rbp-1A8h]
  __int64 v60; // [rsp+48h] [rbp-1A8h]
  __int64 v61; // [rsp+48h] [rbp-1A8h]
  _QWORD *v62; // [rsp+50h] [rbp-1A0h] BYREF
  __int64 v63; // [rsp+58h] [rbp-198h]
  _QWORD v64[4]; // [rsp+60h] [rbp-190h] BYREF
  __int64 *v65; // [rsp+80h] [rbp-170h] BYREF
  __int64 v66; // [rsp+88h] [rbp-168h]
  _QWORD v67[4]; // [rsp+90h] [rbp-160h] BYREF
  _BYTE *v68; // [rsp+B0h] [rbp-140h] BYREF
  __int64 v69; // [rsp+B8h] [rbp-138h]
  _BYTE v70[304]; // [rsp+C0h] [rbp-130h] BYREF

  v9 = v64;
  v13 = *(unsigned int *)(a4 + 32);
  v62 = v64;
  v63 = 0x400000000LL;
  if ( v13 )
  {
    if ( v13 > 4 )
    {
      sub_C8D5F0(&v62, v64, v13, 8);
      v9 = &v62[(unsigned int)v63];
      for ( i = &v62[v13]; i != v9; ++v9 )
      {
LABEL_4:
        if ( v9 )
          *v9 = 0;
      }
    }
    else
    {
      i = &v64[v13];
      if ( i != v64 )
        goto LABEL_4;
    }
    LODWORD(v63) = v13;
  }
  v15 = v67;
  v16 = 0x400000000LL;
  j = v67;
  v65 = v67;
  v66 = 0x400000000LL;
  if ( a6 )
  {
    if ( a6 > 4 )
    {
      v16 = (__int64)v67;
      sub_C8D5F0(&v65, v67, a6, 8);
      v15 = &v65[(unsigned int)v66];
      for ( j = &v65[a6]; j != v15; ++v15 )
      {
LABEL_11:
        if ( v15 )
          *v15 = 0;
      }
    }
    else
    {
      j = &v67[a6];
      if ( j != v67 )
        goto LABEL_11;
    }
    LODWORD(v66) = a6;
  }
  v55 = *(_QWORD **)(a4 + 24);
  if ( a3 != 228 )
  {
    if ( a3 > 0xE4 )
    {
      if ( a3 - 3480 <= 3 )
      {
        v27 = (_BYTE *)*a5;
        if ( *(_BYTE *)*a5 == 17 )
        {
          v46 = (_BYTE *)a4;
          v47 = (_QWORD *)*((_QWORD *)v27 + 3);
          v48 = *(unsigned int *)(a4 + 32);
          if ( *((_DWORD *)v27 + 8) > 0x40u )
            v47 = (_QWORD *)*v47;
          v68 = v70;
          v69 = 0x1000000000LL;
          if ( (_DWORD)v48 )
          {
            for ( k = 0; k != v48; ++k )
            {
              if ( (unsigned __int64)v47 > k )
                v50 = sub_AD6400(v55);
              else
                v50 = sub_AD6450(v55, v46, j);
              j = (__int64 *)(unsigned int)v69;
              if ( (unsigned __int64)(unsigned int)v69 + 1 > HIDWORD(v69) )
              {
                v46 = v70;
                v61 = v50;
                sub_C8D5F0(&v68, v70, (unsigned int)v69 + 1LL, 8);
                j = (__int64 *)(unsigned int)v69;
                v50 = v61;
              }
              *(_QWORD *)&v68[8 * (_QWORD)j] = v50;
              LODWORD(v69) = v69 + 1;
            }
            goto LABEL_97;
          }
LABEL_101:
          v51 = v70;
          v16 = 0;
          goto LABEL_98;
        }
LABEL_34:
        v19 = 0;
        goto LABEL_35;
      }
    }
    else if ( a3 == 185 )
    {
      v18 = *a5;
      v19 = 0;
      v20 = a5[1];
      if ( *(_BYTE *)*a5 != 17 )
        goto LABEL_35;
      if ( *(_BYTE *)v20 == 17 )
      {
        v21 = (_BYTE *)a4;
        v22 = *(_QWORD **)(v18 + 24);
        v23 = *(unsigned int *)(a4 + 32);
        if ( *(_DWORD *)(v18 + 32) > 0x40u )
          v22 = (_QWORD *)*v22;
        v24 = *(_QWORD **)(v20 + 24);
        if ( *(_DWORD *)(v20 + 32) > 0x40u )
          v24 = (_QWORD *)*v24;
        v68 = v70;
        v69 = 0x1000000000LL;
        if ( (_DWORD)v23 )
        {
          v25 = (_QWORD *)((char *)v22 + v23);
          do
          {
            if ( v24 > v22 )
              v26 = sub_AD6400(v55);
            else
              v26 = sub_AD6450(v55, v21, v20);
            v20 = (unsigned int)v69;
            if ( (unsigned __int64)(unsigned int)v69 + 1 > HIDWORD(v69) )
            {
              v21 = v70;
              v60 = v26;
              sub_C8D5F0(&v68, v70, (unsigned int)v69 + 1LL, 8);
              v20 = (unsigned int)v69;
              v26 = v60;
            }
            v22 = (_QWORD *)((char *)v22 + 1);
            *(_QWORD *)&v68[8 * v20] = v26;
            LODWORD(v69) = v69 + 1;
          }
          while ( v25 != v22 );
LABEL_97:
          v51 = v68;
          v16 = (unsigned int)v69;
          goto LABEL_98;
        }
        goto LABEL_101;
      }
      goto LABEL_34;
    }
    v57 = 0;
    v52 = *(unsigned int *)(a4 + 32);
    if ( !(_DWORD)v52 )
    {
LABEL_49:
      v16 = (unsigned int)v63;
      v19 = sub_AD3730(v62, (unsigned int)v63);
      goto LABEL_35;
    }
    while ( 1 )
    {
      v29 = 0;
      if ( (_DWORD)a6 )
        break;
LABEL_47:
      v16 = a2;
      v32 = sub_978F70(a1, a2, a3, v55, v65, (unsigned int)v66, a8, a9);
      if ( !v32 )
        goto LABEL_34;
      v62[v57++] = v32;
      if ( v52 == v57 )
        goto LABEL_49;
    }
    while ( 1 )
    {
      while ( 1 )
      {
        v30 = v29;
        if ( !(unsigned __int8)sub_9B75A0(a3, (unsigned int)v29, 0) )
          break;
        v65[v29] = a5[v29];
        if ( ++v29 == (unsigned int)a6 )
          goto LABEL_47;
      }
      v16 = (unsigned int)v57;
      v31 = sub_AD69F0(a5[v29], (unsigned int)v57);
      if ( !v31 )
        goto LABEL_34;
      ++v29;
      v65[v30] = v31;
      if ( v29 == (unsigned int)a6 )
        goto LABEL_47;
    }
  }
  v59 = a5[2];
  v58 = a5[3];
  v19 = sub_9718F0(*a5, a4, a7);
  v68 = v70;
  v69 = 0x2000000000LL;
  v56 = *(_DWORD *)(a4 + 32);
  if ( !v56 )
  {
    v51 = v70;
    v16 = 0;
    goto LABEL_98;
  }
  for ( m = 0; m != v56; ++m )
  {
    v37 = (_BYTE *)sub_AD69F0(v59, m);
    if ( !v37 )
      break;
    v16 = m;
    v38 = sub_AD69F0(v58, m);
    if ( v19 )
    {
      v16 = m;
      v34 = sub_AD69F0(v19, m);
      if ( (unsigned __int8)(*v37 - 12) > 1u )
      {
        if ( (unsigned __int8)sub_AC30F0(v37) )
          goto LABEL_54;
        goto LABEL_69;
      }
      if ( !v38 )
      {
        if ( !v34 )
          goto LABEL_80;
        v44 = (unsigned int)v69;
        v45 = (unsigned int)v69 + 1LL;
        if ( v45 > HIDWORD(v69) )
        {
          v16 = (__int64)v70;
          sub_C8D5F0(&v68, v70, v45, 8);
          v44 = (unsigned int)v69;
        }
        *(_QWORD *)&v68[8 * v44] = v34;
        LODWORD(v69) = v69 + 1;
        if ( (unsigned __int8)sub_AC30F0(v37) )
          goto LABEL_80;
        goto LABEL_69;
      }
    }
    else
    {
      v39 = (unsigned __int8)*v37;
      if ( v39 != 12 && v39 != 13 )
      {
        if ( !(unsigned __int8)sub_AC30F0(v37) )
        {
          sub_AD7A80(v37);
          goto LABEL_81;
        }
LABEL_54:
        if ( !v38 )
          goto LABEL_80;
LABEL_55:
        v35 = (unsigned int)v69;
        v36 = (unsigned int)v69 + 1LL;
        if ( v36 > HIDWORD(v69) )
        {
          sub_C8D5F0(&v68, v70, v36, 8);
          v35 = (unsigned int)v69;
        }
        *(_QWORD *)&v68[8 * v35] = v38;
        LODWORD(v69) = v69 + 1;
        continue;
      }
      if ( !v38 )
        goto LABEL_80;
      v34 = 0;
    }
    v40 = (unsigned int)v69;
    v41 = (unsigned int)v69 + 1LL;
    if ( v41 > HIDWORD(v69) )
    {
      v16 = (__int64)v70;
      sub_C8D5F0(&v68, v70, v41, 8);
      v40 = (unsigned int)v69;
    }
    *(_QWORD *)&v68[8 * v40] = v38;
    LODWORD(v69) = v69 + 1;
    if ( (unsigned __int8)sub_AC30F0(v37) )
      goto LABEL_55;
LABEL_69:
    if ( (unsigned __int8)sub_AD7A80(v37) != 1 || !v34 )
      goto LABEL_80;
    v42 = (unsigned int)v69;
    v43 = (unsigned int)v69 + 1LL;
    if ( v43 > HIDWORD(v69) )
    {
      sub_C8D5F0(&v68, v70, v43, 8);
      v42 = (unsigned int)v69;
    }
    *(_QWORD *)&v68[8 * v42] = v34;
    LODWORD(v69) = v69 + 1;
  }
  v51 = v68;
  v16 = (unsigned int)v69;
  if ( (_DWORD)v69 != *(_DWORD *)(a4 + 32) )
  {
LABEL_80:
    v19 = 0;
    goto LABEL_81;
  }
LABEL_98:
  v19 = sub_AD3730(v51, v16);
LABEL_81:
  if ( v68 != v70 )
    _libc_free(v68, v16);
LABEL_35:
  if ( v65 != v67 )
    _libc_free(v65, v16);
  if ( v62 != v64 )
    _libc_free(v62, v16);
  return v19;
}
