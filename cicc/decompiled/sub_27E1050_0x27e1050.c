// Function: sub_27E1050
// Address: 0x27e1050
//
__int64 __fastcall sub_27E1050(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // r8
  __int64 v6; // r9
  __int64 v7; // rax
  __int64 i; // rbx
  __int64 v9; // rax
  unsigned __int64 v10; // rdx
  __int64 v11; // rax
  unsigned __int64 v12; // rdx
  __int64 v13; // rax
  char *v14; // rsi
  __int64 v15; // rcx
  __int64 v16; // rax
  char *v17; // rdi
  char *v18; // rax
  char *v19; // rax
  char *v20; // rbx
  __int64 *v21; // r15
  __int64 v22; // rbx
  __int64 *v23; // r12
  __int64 v24; // rax
  __int64 v25; // rbx
  __int64 *v26; // rbx
  __int64 *j; // rbx
  char *v28; // rax
  char *v29; // rbx
  unsigned __int8 *v30; // rax
  size_t v31; // rdx
  unsigned int v32; // esi
  int v33; // edx
  __int64 v34; // rax
  _QWORD *v35; // rbx
  int v36; // ecx
  __int64 v37; // rdx
  unsigned __int64 *v38; // r12
  __int64 v39; // rdx
  _QWORD *v40; // rbx
  __int64 v41; // r8
  unsigned int v42; // ecx
  _QWORD *v43; // rdx
  __int64 v44; // rdi
  int k; // eax
  __int64 v46; // rsi
  int v48; // r11d
  int v49; // edx
  int v50; // ecx
  int v51; // edi
  __int64 v52; // r8
  _QWORD *v53; // rsi
  unsigned int v54; // edx
  __int64 v55; // r10
  int v56; // ecx
  int v57; // edi
  __int64 v58; // r8
  unsigned int v59; // edx
  __int64 v60; // r10
  __int64 v62; // [rsp+30h] [rbp-1C0h]
  unsigned __int8 *v64; // [rsp+50h] [rbp-1A0h]
  __int64 v65; // [rsp+58h] [rbp-198h]
  _QWORD v66[2]; // [rsp+68h] [rbp-188h] BYREF
  __int64 v67; // [rsp+78h] [rbp-178h]
  __int64 v68; // [rsp+80h] [rbp-170h]
  char *v69; // [rsp+90h] [rbp-160h] BYREF
  __int64 v70; // [rsp+98h] [rbp-158h]
  _BYTE v71[32]; // [rsp+A0h] [rbp-150h] BYREF
  void *dest; // [rsp+C0h] [rbp-130h] BYREF
  __int64 v73; // [rsp+C8h] [rbp-128h]
  _BYTE v74[32]; // [rsp+D0h] [rbp-120h] BYREF
  __int64 v75[8]; // [rsp+F0h] [rbp-100h] BYREF
  _BYTE *v76; // [rsp+130h] [rbp-C0h] BYREF
  __int64 v77; // [rsp+138h] [rbp-B8h]
  _BYTE v78[176]; // [rsp+140h] [rbp-B0h] BYREF

  sub_11D2BF0((__int64)v75, 0);
  v76 = v78;
  v77 = 0x1000000000LL;
  v69 = v71;
  v70 = 0x400000000LL;
  v73 = 0x400000000LL;
  v7 = *(_QWORD *)(a2 + 56);
  dest = v74;
  v65 = v7;
  v62 = a2 + 48;
  if ( v7 == a2 + 48 )
    goto LABEL_74;
  do
  {
    if ( !v65 )
      BUG();
    for ( i = *(_QWORD *)(v65 - 8); i; i = *(_QWORD *)(i + 8) )
    {
      v11 = *(_QWORD *)(i + 24);
      if ( *(_BYTE *)v11 == 84 )
      {
        if ( a2 != *(_QWORD *)(*(_QWORD *)(v11 - 8)
                             + 32LL * *(unsigned int *)(v11 + 72)
                             + 8LL * (unsigned int)((i - *(_QWORD *)(v11 - 8)) >> 5)) )
          goto LABEL_6;
      }
      else if ( a2 != *(_QWORD *)(v11 + 40) )
      {
LABEL_6:
        v9 = (unsigned int)v77;
        v10 = (unsigned int)v77 + 1LL;
        if ( v10 > HIDWORD(v77) )
        {
          sub_C8D5F0((__int64)&v76, v78, v10, 8u, v5, v6);
          v9 = (unsigned int)v77;
        }
        *(_QWORD *)&v76[8 * v9] = i;
        LODWORD(v77) = v77 + 1;
      }
    }
    v64 = (unsigned __int8 *)(v65 - 24);
    sub_AE7A40((__int64)&v69, (_BYTE *)(v65 - 24), (__int64)&dest);
    v12 = (unsigned __int64)v69;
    v13 = 8LL * (unsigned int)v70;
    v14 = &v69[v13];
    v15 = v13 >> 3;
    v16 = v13 >> 5;
    if ( v16 )
    {
      v17 = v69;
      v18 = &v69[32 * v16];
      while ( a2 != *(_QWORD *)(*(_QWORD *)v17 + 40LL) )
      {
        if ( a2 == *(_QWORD *)(*((_QWORD *)v17 + 1) + 40LL) )
        {
          v17 += 8;
          goto LABEL_20;
        }
        if ( a2 == *(_QWORD *)(*((_QWORD *)v17 + 2) + 40LL) )
        {
          v17 += 16;
          goto LABEL_20;
        }
        if ( a2 == *(_QWORD *)(*((_QWORD *)v17 + 3) + 40LL) )
        {
          v17 += 24;
          goto LABEL_20;
        }
        v17 += 32;
        if ( v18 == v17 )
        {
          v15 = (v14 - v17) >> 3;
          goto LABEL_89;
        }
      }
      goto LABEL_20;
    }
    v17 = v69;
LABEL_89:
    if ( v15 == 2 )
      goto LABEL_95;
    if ( v15 != 3 )
    {
      if ( v15 != 1 )
      {
LABEL_92:
        v20 = v14;
        goto LABEL_27;
      }
LABEL_97:
      v20 = v14;
      if ( a2 != *(_QWORD *)(*(_QWORD *)v17 + 40LL) )
        goto LABEL_27;
      goto LABEL_20;
    }
    if ( a2 != *(_QWORD *)(*(_QWORD *)v17 + 40LL) )
    {
      v17 += 8;
LABEL_95:
      if ( a2 != *(_QWORD *)(*(_QWORD *)v17 + 40LL) )
      {
        v17 += 8;
        goto LABEL_97;
      }
    }
LABEL_20:
    if ( v14 == v17 )
      goto LABEL_92;
    v19 = v17 + 8;
    if ( v14 == v17 + 8 )
    {
      v20 = v17;
    }
    else
    {
      do
      {
        if ( a2 != *(_QWORD *)(*(_QWORD *)v19 + 40LL) )
        {
          *(_QWORD *)v17 = *(_QWORD *)v19;
          v17 += 8;
        }
        v19 += 8;
      }
      while ( v14 != v19 );
      v12 = (unsigned __int64)v69;
      v5 = &v69[8 * (unsigned int)v70] - v14;
      v20 = &v17[v5];
      if ( v14 != &v69[8 * (unsigned int)v70] )
      {
        memmove(v17, v14, &v69[8 * (unsigned int)v70] - v14);
        v12 = (unsigned __int64)v69;
      }
    }
LABEL_27:
    v21 = (__int64 *)dest;
    LODWORD(v70) = (__int64)&v20[-v12] >> 3;
    v22 = 8LL * (unsigned int)v73;
    v23 = (__int64 *)((char *)dest + v22);
    v24 = v22 >> 3;
    v25 = v22 >> 5;
    if ( v25 )
    {
      v26 = (__int64 *)((char *)dest + 32 * v25);
      while ( a2 != sub_B140B0(*v21) )
      {
        if ( a2 == sub_B140B0(v21[1]) )
        {
          ++v21;
          goto LABEL_34;
        }
        if ( a2 == sub_B140B0(v21[2]) )
        {
          v21 += 2;
          goto LABEL_34;
        }
        if ( a2 == sub_B140B0(v21[3]) )
        {
          v21 += 3;
          goto LABEL_34;
        }
        v21 += 4;
        if ( v26 == v21 )
        {
          v24 = v23 - v21;
          goto LABEL_78;
        }
      }
      goto LABEL_34;
    }
LABEL_78:
    if ( v24 == 2 )
      goto LABEL_101;
    if ( v24 == 3 )
    {
      if ( a2 == sub_B140B0(*v21) )
        goto LABEL_34;
      ++v21;
LABEL_101:
      if ( a2 == sub_B140B0(*v21) )
        goto LABEL_34;
      ++v21;
      goto LABEL_103;
    }
    if ( v24 != 1 )
      goto LABEL_81;
LABEL_103:
    if ( a2 != sub_B140B0(*v21) )
    {
LABEL_81:
      v21 = v23;
      goto LABEL_39;
    }
LABEL_34:
    if ( v23 != v21 )
    {
      for ( j = v21 + 1; v23 != j; ++j )
      {
        if ( a2 != sub_B140B0(*j) )
          *v21++ = *j;
      }
    }
LABEL_39:
    v28 = (char *)dest;
    v29 = (char *)((_BYTE *)dest + 8 * (unsigned int)v73 - (_BYTE *)v23);
    if ( v23 != (__int64 *)((char *)dest + 8 * (unsigned int)v73) )
    {
      memmove(v21, v23, (_BYTE *)dest + 8 * (unsigned int)v73 - (_BYTE *)v23);
      v28 = (char *)dest;
    }
    LODWORD(v73) = (&v29[(_QWORD)v21] - v28) >> 3;
    if ( (unsigned int)v73 | (unsigned int)v70 | (unsigned int)v77 )
    {
      v30 = (unsigned __int8 *)sub_BD5D20((__int64)v64);
      sub_11D2C80(v75, *(_QWORD *)(v65 - 16), v30, v31);
      sub_11D33F0(v75, a2, (__int64)v64);
      v67 = v65 - 24;
      v66[0] = 2;
      v66[1] = 0;
      if ( v65 != -4072 && v65 != -8168 )
        sub_BD73F0((__int64)v66);
      v32 = *(_DWORD *)(a4 + 24);
      v68 = a4;
      if ( v32 )
      {
        v41 = *(_QWORD *)(a4 + 8);
        v34 = v67;
        v42 = (v32 - 1) & (((unsigned int)v67 >> 9) ^ ((unsigned int)v67 >> 4));
        v43 = (_QWORD *)(v41 + ((unsigned __int64)v42 << 6));
        v44 = v43[3];
        if ( v67 == v44 )
        {
LABEL_61:
          v40 = v43 + 5;
LABEL_62:
          if ( v34 != 0 && v34 != -4096 && v34 != -8192 )
            sub_BD60C0(v66);
          sub_11D33F0(v75, a3, v40[2]);
          for ( k = v77; (_DWORD)v77; k = v77 )
          {
            v46 = *(_QWORD *)&v76[8 * k - 8];
            LODWORD(v77) = k - 1;
            sub_11D9630(v75, v46);
          }
          if ( (unsigned int)v73 | (unsigned int)v70 )
          {
            sub_11D9740(v75, (__int64)v64, (__int64)&v69);
            sub_11D97F0(v75, v64, (__int64)&dest);
            LODWORD(v70) = 0;
            LODWORD(v73) = 0;
          }
          goto LABEL_69;
        }
        v48 = 1;
        v35 = 0;
        while ( v44 != -4096 )
        {
          if ( !v35 && v44 == -8192 )
            v35 = v43;
          v42 = (v32 - 1) & (v48 + v42);
          v43 = (_QWORD *)(v41 + ((unsigned __int64)v42 << 6));
          v44 = v43[3];
          if ( v67 == v44 )
            goto LABEL_61;
          ++v48;
        }
        if ( !v35 )
          v35 = v43;
        ++*(_QWORD *)a4;
        v36 = *(_DWORD *)(a4 + 16) + 1;
        if ( 4 * v36 < 3 * v32 )
        {
          if ( v32 - *(_DWORD *)(a4 + 20) - v36 > v32 >> 3 )
          {
LABEL_50:
            *(_DWORD *)(a4 + 16) = v36;
            if ( v35[3] == -4096 )
            {
              v38 = v35 + 1;
              if ( v34 != -4096 )
              {
LABEL_55:
                v35[3] = v34;
                if ( v34 != 0 && v34 != -4096 && v34 != -8192 )
                  sub_BD6050(v38, v66[0] & 0xFFFFFFFFFFFFFFF8LL);
                v34 = v67;
              }
            }
            else
            {
              --*(_DWORD *)(a4 + 20);
              v37 = v35[3];
              if ( v37 != v34 )
              {
                v38 = v35 + 1;
                if ( v37 != 0 && v37 != -4096 && v37 != -8192 )
                {
                  sub_BD60C0(v35 + 1);
                  v34 = v67;
                }
                goto LABEL_55;
              }
            }
            v39 = v68;
            v35[5] = 6;
            v40 = v35 + 5;
            v40[1] = 0;
            *(v40 - 1) = v39;
            v40[2] = 0;
            goto LABEL_62;
          }
          sub_CF32C0(a4, v32);
          v49 = *(_DWORD *)(a4 + 24);
          if ( !v49 )
            goto LABEL_48;
          v50 = v49 - 1;
          v51 = 1;
          v52 = *(_QWORD *)(a4 + 8);
          v34 = v67;
          v53 = 0;
          v54 = (v49 - 1) & (((unsigned int)v67 >> 9) ^ ((unsigned int)v67 >> 4));
          v35 = (_QWORD *)(v52 + ((unsigned __int64)v54 << 6));
          v55 = v35[3];
          if ( v55 == v67 )
            goto LABEL_49;
          while ( v55 != -4096 )
          {
            if ( !v53 && v55 == -8192 )
              v53 = v35;
            v54 = v50 & (v51 + v54);
            v35 = (_QWORD *)(v52 + ((unsigned __int64)v54 << 6));
            v55 = v35[3];
            if ( v67 == v55 )
              goto LABEL_49;
            ++v51;
          }
          goto LABEL_124;
        }
      }
      else
      {
        ++*(_QWORD *)a4;
      }
      sub_CF32C0(a4, 2 * v32);
      v33 = *(_DWORD *)(a4 + 24);
      if ( !v33 )
      {
LABEL_48:
        v34 = v67;
        v35 = 0;
LABEL_49:
        v36 = *(_DWORD *)(a4 + 16) + 1;
        goto LABEL_50;
      }
      v56 = v33 - 1;
      v57 = 1;
      v58 = *(_QWORD *)(a4 + 8);
      v34 = v67;
      v53 = 0;
      v59 = (v33 - 1) & (((unsigned int)v67 >> 9) ^ ((unsigned int)v67 >> 4));
      v35 = (_QWORD *)(v58 + ((unsigned __int64)v59 << 6));
      v60 = v35[3];
      if ( v60 == v67 )
        goto LABEL_49;
      while ( v60 != -4096 )
      {
        if ( !v53 && v60 == -8192 )
          v53 = v35;
        v59 = v56 & (v57 + v59);
        v35 = (_QWORD *)(v58 + ((unsigned __int64)v59 << 6));
        v60 = v35[3];
        if ( v67 == v60 )
          goto LABEL_49;
        ++v57;
      }
LABEL_124:
      if ( v53 )
        v35 = v53;
      goto LABEL_49;
    }
LABEL_69:
    v65 = *(_QWORD *)(v65 + 8);
  }
  while ( v62 != v65 );
  if ( dest != v74 )
    _libc_free((unsigned __int64)dest);
  if ( v69 != v71 )
    _libc_free((unsigned __int64)v69);
LABEL_74:
  if ( v76 != v78 )
    _libc_free((unsigned __int64)v76);
  return sub_11D2C20(v75);
}
