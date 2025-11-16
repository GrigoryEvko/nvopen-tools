// Function: sub_AD9660
// Address: 0xad9660
//
__int64 __fastcall sub_AD9660(size_t n, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned int v5; // r14d
  __int64 v7; // rdi
  __int64 v8; // rsi
  __int64 v9; // rdx
  _QWORD *v10; // rax
  unsigned int v11; // r13d
  char *v12; // r13
  __int64 v13; // rax
  char *v14; // rsi
  __int64 v15; // rax
  void *v16; // rdi
  __int64 v17; // r12
  char v19; // al
  __int64 v20; // rsi
  char *v21; // rdx
  _QWORD *v22; // r13
  __int64 v23; // r15
  char *v24; // r14
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rsi
  unsigned int v29; // r15d
  __int16 v30; // r13
  __int64 v31; // rdx
  __int64 v32; // rax
  char v33; // al
  __int64 *v34; // rdx
  _QWORD *v35; // r13
  __int64 *v36; // rax
  __int64 v37; // rax
  __int64 v38; // rax
  __int64 *v39; // rax
  __int64 v40; // rax
  __int64 v41; // rax
  unsigned int v42; // ebx
  __int64 v43; // r15
  __int64 *v44; // rax
  __int64 v45; // rax
  __int16 v46; // r13
  __int16 *v47; // rax
  char *v48; // rax
  unsigned int v49; // ebx
  int v50; // r15d
  __int64 *v51; // rax
  __int64 v52; // rax
  __int16 *v53; // rax
  char *v54; // rax
  __int16 *v55; // rax
  __int64 v56; // rdx
  __int16 *v57; // rcx
  __int16 *v58; // rax
  __int16 *v59; // rcx
  __int64 *v60; // rax
  __int64 *v61; // rax
  int *v62; // rax
  int *v63; // rdx
  __int64 *v64; // rax
  __int64 *v65; // rdx
  int *v66; // [rsp+10h] [rbp-D0h] BYREF
  unsigned int v67; // [rsp+18h] [rbp-C8h]
  void *v68; // [rsp+20h] [rbp-C0h] BYREF
  __int64 v69; // [rsp+28h] [rbp-B8h]
  __int64 v70; // [rsp+30h] [rbp-B0h] BYREF
  _WORD s[84]; // [rsp+38h] [rbp-A8h] BYREF

  v5 = n;
  if ( *(_BYTE *)a2 == 17 )
  {
    v7 = *(_QWORD *)(a2 + 8);
    v8 = 8;
    if ( (unsigned __int8)sub_BCAC40(v7, 8) )
    {
      v10 = *(_QWORD **)(a2 + 24);
      if ( *(_DWORD *)(a2 + 32) > 0x40u )
        v10 = (_QWORD *)*v10;
      v11 = (unsigned __int8)v10;
      v68 = s;
      v69 = 0;
      v70 = 16;
      if ( v5 > 0x10 )
      {
        sub_C8D290(&v68, s, v5, 1);
        v8 = v11;
        memset(v68, v11, v5);
        v69 = v5;
        v12 = (char *)v68;
      }
      else
      {
        if ( v5 )
        {
          v8 = (unsigned __int8)v10;
          memset(s, (unsigned __int8)v10, v5);
        }
        v69 = v5;
        v12 = (char *)s;
      }
      v13 = sub_BD5C60(a2, v8, v9);
      v14 = v12;
      v15 = sub_AC99A0(v13, v12, v5);
      v16 = v68;
      v17 = v15;
      if ( v68 == s )
        return v17;
    }
    else
    {
      v20 = 16;
      if ( (unsigned __int8)sub_BCAC40(*(_QWORD *)(a2 + 8), 16) )
      {
        v22 = *(_QWORD **)(a2 + 24);
        if ( *(_DWORD *)(a2 + 32) > 0x40u )
          v22 = (_QWORD *)*v22;
        v23 = v5;
        v69 = 0;
        v68 = s;
        v70 = 16;
        if ( v5 > 0x10 )
        {
          v20 = (__int64)s;
          sub_C8D290(&v68, s, v5, 2);
          v54 = (char *)v68;
          v21 = (char *)v68 + 2 * v5;
          do
          {
            *(_WORD *)v54 = (_WORD)v22;
            v54 += 2;
          }
          while ( v21 != v54 );
          v69 = v5;
          v24 = (char *)v68;
        }
        else
        {
          v24 = (char *)s;
          if ( v23 )
          {
            v21 = (char *)&s[v23];
            v48 = (char *)s;
            do
            {
              *(_WORD *)v48 = (_WORD)v22;
              v48 += 2;
            }
            while ( v21 != v48 );
            v24 = (char *)v68;
          }
          v69 = v23;
        }
        v25 = sub_BD5C60(a2, v20, v21);
        v14 = v24;
        v26 = sub_AC99E0(v25, v24, v23);
        v16 = v68;
        v17 = v26;
        if ( v68 == s )
          return v17;
      }
      else
      {
        v33 = sub_BCAC40(*(_QWORD *)(a2 + 8), 32);
        v35 = *(_QWORD **)(a2 + 24);
        if ( v33 )
        {
          if ( *(_DWORD *)(a2 + 32) > 0x40u )
            v35 = (_QWORD *)*v35;
          v68 = &v70;
          v69 = 0x1000000000LL;
          if ( v5 > 0x10 )
          {
            sub_C8D5F0(&v68, &v70, v5, 4);
            v61 = (__int64 *)v68;
            v34 = (__int64 *)((char *)v68 + 4 * v5);
            do
            {
              *(_DWORD *)v61 = (_DWORD)v35;
              v61 = (__int64 *)((char *)v61 + 4);
            }
            while ( v34 != v61 );
            LODWORD(v69) = v5;
            v14 = (char *)v68;
          }
          else
          {
            v14 = (char *)&v70;
            if ( v5 )
            {
              v34 = (__int64 *)&s[2 * v5 - 4];
              v36 = &v70;
              do
              {
                *(_DWORD *)v36 = (_DWORD)v35;
                v36 = (__int64 *)((char *)v36 + 4);
              }
              while ( v34 != v36 );
              v14 = (char *)v68;
            }
            LODWORD(v69) = v5;
          }
          v37 = sub_BD5C60(a2, v14, v34);
          v38 = sub_AC9A10(v37, v14, v5);
          v16 = v68;
          v17 = v38;
          if ( v68 == &v70 )
            return v17;
        }
        else
        {
          if ( *(_DWORD *)(a2 + 32) > 0x40u )
            v35 = (_QWORD *)*v35;
          v68 = &v70;
          v69 = 0x1000000000LL;
          if ( v5 > 0x10 )
          {
            sub_C8D5F0(&v68, &v70, v5, 8);
            v60 = (__int64 *)v68;
            v34 = (__int64 *)((char *)v68 + 8 * v5);
            do
              *v60++ = (__int64)v35;
            while ( v34 != v60 );
            LODWORD(v69) = v5;
            v14 = (char *)v68;
          }
          else
          {
            v14 = (char *)&v70;
            if ( v5 )
            {
              v34 = (__int64 *)&s[4 * v5 - 4];
              v39 = &v70;
              do
                *v39++ = (__int64)v35;
              while ( v34 != v39 );
              v14 = (char *)v68;
            }
            LODWORD(v69) = v5;
          }
          v40 = sub_BD5C60(a2, v14, v34);
          v41 = sub_AC9A50(v40, v14, v5);
          v16 = v68;
          v17 = v41;
          if ( v68 == &v70 )
            return v17;
        }
      }
    }
LABEL_10:
    _libc_free(v16, v14);
    return v17;
  }
  if ( *(_BYTE *)a2 == 18 )
  {
    v19 = *(_BYTE *)(*(_QWORD *)(a2 + 8) + 8LL);
    if ( v19 )
    {
      if ( v19 != 1 )
      {
        if ( v19 == 2 )
        {
          sub_9875A0((__int64)&v66, (_QWORD *)(a2 + 24), a3, a4, a5);
          v49 = v67;
          if ( v67 > 0x40 )
          {
            v50 = -1;
            if ( v49 - (unsigned int)sub_C444A0(&v66) <= 0x40 )
              v50 = *v66;
          }
          else
          {
            v50 = (int)v66;
          }
          v68 = &v70;
          v69 = 0x1000000000LL;
          if ( (unsigned int)n > 0x10 )
          {
            sub_C8D5F0(&v68, &v70, (unsigned int)n, 4);
            v62 = (int *)v68;
            v63 = (int *)((char *)v68 + 4 * (unsigned int)n);
            do
              *v62++ = v50;
            while ( v63 != v62 );
          }
          else if ( (_DWORD)n )
          {
            v51 = &v70;
            do
            {
              *(_DWORD *)v51 = v50;
              v51 = (__int64 *)((char *)v51 + 4);
            }
            while ( &s[2 * (unsigned int)n - 4] != (_WORD *)v51 );
          }
          LODWORD(v69) = n;
          sub_969240((__int64 *)&v66);
          v14 = (char *)v68;
          v52 = sub_AC9AC0(*(_QWORD *)(a2 + 8), (char *)v68, (unsigned int)v69);
          v16 = v68;
          v17 = v52;
          if ( v68 == &v70 )
            return v17;
          goto LABEL_10;
        }
        if ( v19 == 3 )
        {
          sub_9875A0((__int64)&v66, (_QWORD *)(a2 + 24), a3, a4, a5);
          v42 = v67;
          if ( v67 > 0x40 )
          {
            v43 = -1;
            if ( v42 - (unsigned int)sub_C444A0(&v66) <= 0x40 )
              v43 = *(_QWORD *)v66;
          }
          else
          {
            v43 = (__int64)v66;
          }
          v68 = &v70;
          v69 = 0x1000000000LL;
          if ( (unsigned int)n > 0x10 )
          {
            sub_C8D5F0(&v68, &v70, (unsigned int)n, 8);
            v64 = (__int64 *)v68;
            v65 = (__int64 *)((char *)v68 + 8 * (unsigned int)n);
            do
              *v64++ = v43;
            while ( v65 != v64 );
          }
          else if ( (_DWORD)n )
          {
            v44 = &v70;
            do
              *v44++ = v43;
            while ( &s[4 * (unsigned int)n - 4] != (_WORD *)v44 );
          }
          LODWORD(v69) = n;
          sub_969240((__int64 *)&v66);
          v14 = (char *)v68;
          v45 = sub_AC9AF0(*(_QWORD *)(a2 + 8), (char *)v68, (unsigned int)v69);
          v16 = v68;
          v17 = v45;
          if ( v68 == &v70 )
            return v17;
          goto LABEL_10;
        }
        goto LABEL_17;
      }
      sub_9875A0((__int64)&v66, (_QWORD *)(a2 + 24), a3, a4, a5);
      v29 = v67;
      if ( v67 > 0x40 )
      {
        v46 = -1;
        if ( v29 - (unsigned int)sub_C444A0(&v66) <= 0x40 )
          v46 = *(_WORD *)v66;
      }
      else
      {
        v46 = (__int16)v66;
      }
      v31 = (unsigned int)n;
      v69 = 0;
      v68 = s;
      v70 = 16;
      if ( (unsigned int)n <= 0x10 )
      {
        if ( (_DWORD)n )
        {
          v47 = s;
          do
            *v47++ = v46;
          while ( &s[(unsigned int)n] != v47 );
LABEL_76:
          v29 = v67;
          goto LABEL_33;
        }
        goto LABEL_33;
      }
      sub_C8D290(&v68, s, (unsigned int)n, 2);
      v58 = (__int16 *)v68;
      v56 = (unsigned int)n;
      v59 = (__int16 *)((char *)v68 + 2 * (unsigned int)n);
      do
        *v58++ = v46;
      while ( v59 != v58 );
    }
    else
    {
      v27 = sub_C33340(n, a2, a3, a4, a5);
      v28 = a2 + 24;
      if ( *(_QWORD *)(a2 + 24) == v27 )
        sub_C3E660(&v66, v28);
      else
        sub_C3A850(&v66, v28);
      v29 = v67;
      if ( v67 <= 0x40 )
      {
        v30 = (__int16)v66;
      }
      else
      {
        v30 = -1;
        if ( v29 - (unsigned int)sub_C444A0(&v66) <= 0x40 )
          v30 = *(_WORD *)v66;
      }
      v31 = (unsigned int)n;
      v69 = 0;
      v68 = s;
      v70 = 16;
      if ( (unsigned int)n <= 0x10 )
      {
        if ( (_DWORD)n )
        {
          v53 = s;
          do
            *v53++ = v30;
          while ( &s[(unsigned int)n] != v53 );
          goto LABEL_76;
        }
LABEL_33:
        v69 = v31;
LABEL_34:
        if ( v29 > 0x40 && v66 )
          j_j___libc_free_0_0(v66);
        v14 = (char *)v68;
        v32 = sub_AC9A90(*(_QWORD *)(a2 + 8), (char *)v68, v69);
        v16 = v68;
        v17 = v32;
        if ( v68 == s )
          return v17;
        goto LABEL_10;
      }
      sub_C8D290(&v68, s, (unsigned int)n, 2);
      v55 = (__int16 *)v68;
      v56 = (unsigned int)n;
      v57 = (__int16 *)((char *)v68 + 2 * (unsigned int)n);
      do
        *v55++ = v30;
      while ( v57 != v55 );
    }
    v69 = v56;
    v29 = v67;
    goto LABEL_34;
  }
LABEL_17:
  LODWORD(v68) = n;
  BYTE4(v68) = 0;
  return sub_AD5E10((__int64)v68, (unsigned __int8 *)a2);
}
