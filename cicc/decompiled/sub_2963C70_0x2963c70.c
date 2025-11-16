// Function: sub_2963C70
// Address: 0x2963c70
//
void __fastcall sub_2963C70(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 *a5, __int64 a6)
{
  __int64 v6; // r13
  __int64 *v10; // r10
  __int64 v11; // rax
  __int64 *v12; // rcx
  __int64 v13; // rbx
  _QWORD *v14; // rsi
  unsigned int v15; // r8d
  unsigned int v16; // r12d
  __int64 v17; // rdi
  unsigned int v18; // edx
  __int64 *v19; // rax
  __int64 v20; // r9
  _QWORD *v21; // rdx
  _QWORD *v22; // rax
  __int64 v23; // rsi
  __int64 *v24; // r12
  __int64 *v25; // rbx
  __int64 v26; // r8
  __int64 v27; // r9
  __int64 *v28; // r12
  __int64 *v29; // rbx
  __int64 v30; // rsi
  _QWORD *v31; // rcx
  _QWORD *v32; // rax
  __int64 v33; // rdi
  __int64 v34; // rdx
  __int64 v35; // rcx
  __int64 v36; // r14
  int v37; // r9d
  char *v38; // rsi
  unsigned int v39; // edi
  char *v40; // rax
  __int64 v41; // rdx
  _QWORD *v42; // rax
  _BYTE *v43; // rax
  _QWORD *v44; // rbx
  __int64 v45; // rax
  __int64 v46; // r13
  __int64 v47; // rbx
  __int64 v48; // rax
  __int64 *v49; // rbx
  __int64 v50; // rsi
  _QWORD *v51; // rax
  _QWORD *v52; // rdx
  _QWORD *v53; // rax
  _QWORD *v54; // rdx
  __int64 *v55; // rax
  __int64 v56; // r8
  _QWORD *v57; // rsi
  __int64 *v58; // rax
  __int64 v59; // r8
  _QWORD *v60; // rax
  __int64 *v61; // rax
  __int64 v62; // rsi
  __int64 *v63; // rax
  int v64; // eax
  unsigned int v65; // r8d
  unsigned int v66; // edx
  __int64 *v67; // rax
  __int64 v68; // rsi
  __int64 v69; // rbx
  __int64 v70; // rcx
  int v71; // edx
  int v72; // edi
  int v73; // eax
  int v74; // ecx
  _QWORD *v79; // [rsp+28h] [rbp-98h]
  char *v80; // [rsp+38h] [rbp-88h]
  __int64 v81; // [rsp+48h] [rbp-78h]
  int v82; // [rsp+48h] [rbp-78h]
  __int64 v83; // [rsp+50h] [rbp-70h] BYREF
  char *v84; // [rsp+58h] [rbp-68h] BYREF
  __int64 *v85; // [rsp+60h] [rbp-60h] BYREF
  __int64 v86; // [rsp+68h] [rbp-58h]
  _BYTE v87[80]; // [rsp+70h] [rbp-50h] BYREF

  v6 = *(_QWORD *)a1;
  if ( !*(_QWORD *)a1 )
    return;
  v85 = (__int64 *)v87;
  v86 = 0x400000000LL;
  sub_D472F0(a1, (__int64)&v85);
  v10 = &v85[(unsigned int)v86];
  if ( v85 == v10 )
  {
    v83 = a2;
    v13 = *(_QWORD *)(a4 + 8);
    v15 = *(_DWORD *)(a4 + 24);
LABEL_121:
    if ( v15 )
    {
      v65 = v15 - 1;
      v66 = v65 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v67 = (__int64 *)(v13 + 16LL * v66);
      v68 = *v67;
      if ( a2 == *v67 )
      {
LABEL_123:
        *v67 = -8192;
        v79 = 0;
        --*(_DWORD *)(a4 + 16);
        ++*(_DWORD *)(a4 + 20);
        goto LABEL_38;
      }
      v73 = 1;
      while ( v68 != -4096 )
      {
        v74 = v73 + 1;
        v66 = v65 & (v73 + v66);
        v67 = (__int64 *)(v13 + 16LL * v66);
        v68 = *v67;
        if ( a2 == *v67 )
          goto LABEL_123;
        v73 = v74;
      }
    }
    v79 = 0;
LABEL_38:
    v84 = (char *)a1;
    v43 = sub_29578B0(*(_QWORD **)(v6 + 8), *(_QWORD *)(v6 + 16), (__int64 *)&v84);
    v44 = *(_QWORD **)v43;
    sub_D4C9B0(v6 + 8, v43);
    *v44 = 0;
    v84 = (char *)a1;
    if ( v79 )
    {
      *(_QWORD *)a1 = v79;
      sub_D4C980((__int64)(v79 + 1), &v84);
    }
    else
    {
      sub_D4C980(a4 + 32, &v84);
    }
    v81 = a1 + 56;
    v45 = a2;
    v36 = v6;
    v46 = v45;
LABEL_41:
    v24 = *(__int64 **)(v36 + 32);
    v80 = *(char **)(v36 + 40);
    v47 = (v80 - (char *)v24) >> 5;
    v48 = (v80 - (char *)v24) >> 3;
    if ( v47 <= 0 )
      goto LABEL_91;
    v49 = &v24[4 * v47];
    while ( 1 )
    {
      v50 = *v24;
      if ( v46 == *v24 )
        goto LABEL_49;
      if ( *(_BYTE *)(a1 + 84) )
        break;
      if ( sub_C8CA60(v81, v50) )
        goto LABEL_49;
      v56 = v24[1];
      v34 = (__int64)(v24 + 1);
      if ( v46 == v56 )
        goto LABEL_70;
      if ( *(_BYTE *)(a1 + 84) )
      {
        v51 = *(_QWORD **)(a1 + 64);
        v35 = (__int64)&v51[*(unsigned int *)(a1 + 76)];
        if ( v51 != (_QWORD *)v35 )
          goto LABEL_67;
LABEL_102:
        v34 = (__int64)(v24 + 2);
        if ( v46 == v24[2] )
          goto LABEL_70;
        goto LABEL_103;
      }
      v58 = sub_C8CA60(v81, v24[1]);
      v34 = (__int64)(v24 + 1);
      if ( v58 )
        goto LABEL_70;
      v59 = v24[2];
      v34 = (__int64)(v24 + 2);
      if ( v46 == v59 )
        goto LABEL_70;
      if ( *(_BYTE *)(a1 + 84) )
      {
        v57 = *(_QWORD **)(a1 + 64);
        v35 = (__int64)&v57[*(unsigned int *)(a1 + 76)];
        if ( v57 != (_QWORD *)v35 )
          goto LABEL_75;
LABEL_103:
        v34 = (__int64)(v24 + 3);
        if ( v46 == v24[3] )
          goto LABEL_70;
        v24 += 4;
        if ( v24 == v49 )
        {
LABEL_90:
          v48 = (v80 - (char *)v24) >> 3;
LABEL_91:
          if ( v48 == 2 )
            goto LABEL_113;
          if ( v48 == 3 )
          {
            if ( v46 == *v24 || (unsigned __int8)sub_B19060(v81, *v24, v34, v35) )
              goto LABEL_49;
            ++v24;
LABEL_113:
            if ( v46 != *v24 && !(unsigned __int8)sub_B19060(v81, *v24, v34, v35) )
            {
              ++v24;
              goto LABEL_116;
            }
            goto LABEL_49;
          }
          if ( v48 != 1 )
            goto LABEL_94;
LABEL_116:
          if ( v46 != *v24 && !(unsigned __int8)sub_B19060(v81, *v24, v34, v35) )
          {
LABEL_94:
            v24 = (__int64 *)v80;
            goto LABEL_19;
          }
LABEL_49:
          if ( v80 != (char *)v24 )
          {
            v25 = v24 + 1;
            if ( v80 != (char *)(v24 + 1) )
            {
              while ( 1 )
              {
LABEL_51:
                v23 = *v25;
                if ( v46 == *v25 )
                  goto LABEL_18;
                if ( !*(_BYTE *)(a1 + 84) )
                  break;
                v53 = *(_QWORD **)(a1 + 64);
                v54 = &v53[*(unsigned int *)(a1 + 76)];
                if ( v53 == v54 )
                  goto LABEL_17;
                while ( v23 != *v53 )
                {
                  if ( v54 == ++v53 )
                    goto LABEL_17;
                }
                if ( v80 == (char *)++v25 )
                  goto LABEL_19;
              }
              if ( !sub_C8CA60(v81, v23) )
              {
                v23 = *v25;
LABEL_17:
                *v24++ = v23;
              }
LABEL_18:
              if ( v80 == (char *)++v25 )
                goto LABEL_19;
              goto LABEL_51;
            }
          }
LABEL_19:
          sub_295D210(v36 + 32, (char *)v24, v80);
          sub_25DDDB0(v36 + 56, v46);
          v28 = *(__int64 **)(a1 + 32);
          if ( v28 != *(__int64 **)(a1 + 40) )
          {
            v29 = *(__int64 **)(a1 + 40);
            do
            {
              v30 = *v28;
              if ( *(_BYTE *)(v36 + 84) )
              {
                v26 = *(_QWORD *)(v36 + 64);
                v31 = (_QWORD *)(v26 + 8LL * *(unsigned int *)(v36 + 76));
                v32 = (_QWORD *)v26;
                if ( (_QWORD *)v26 != v31 )
                {
                  while ( v30 != *v32 )
                  {
                    if ( v31 == ++v32 )
                      goto LABEL_27;
                  }
                  v33 = (unsigned int)(*(_DWORD *)(v36 + 76) - 1);
                  *(_DWORD *)(v36 + 76) = v33;
                  *v32 = *(_QWORD *)(v26 + 8 * v33);
                  ++*(_QWORD *)(v36 + 56);
                }
              }
              else
              {
                v55 = sub_C8CA60(v36 + 56, v30);
                if ( v55 )
                {
                  *v55 = -2;
                  ++*(_DWORD *)(v36 + 80);
                  ++*(_QWORD *)(v36 + 56);
                }
              }
LABEL_27:
              ++v28;
            }
            while ( v29 != v28 );
          }
          sub_11D1C10(v36, a3, a4, a6, v26, v27);
          sub_F6D150(v36, a3, a4, a5, 1);
          v36 = *(_QWORD *)v36;
          if ( (_QWORD *)v36 == v79 )
          {
            if ( v85 != (__int64 *)v87 )
              _libc_free((unsigned __int64)v85);
            return;
          }
          goto LABEL_41;
        }
      }
      else
      {
        v61 = sub_C8CA60(v81, v24[2]);
        v34 = (__int64)(v24 + 2);
        if ( v61 )
          goto LABEL_70;
        v62 = v24[3];
        v34 = (__int64)(v24 + 3);
        if ( v46 == v62 )
          goto LABEL_70;
        if ( !*(_BYTE *)(a1 + 84) )
        {
          v63 = sub_C8CA60(v81, v62);
          v34 = (__int64)(v24 + 3);
          if ( v63 )
            goto LABEL_70;
          goto LABEL_89;
        }
        v60 = *(_QWORD **)(a1 + 64);
        v35 = (__int64)&v60[*(unsigned int *)(a1 + 76)];
LABEL_83:
        if ( (_QWORD *)v35 != v60 )
        {
          while ( v62 != *v60 )
          {
            if ( ++v60 == (_QWORD *)v35 )
              goto LABEL_89;
          }
LABEL_70:
          v24 = (__int64 *)v34;
          goto LABEL_49;
        }
LABEL_89:
        v24 += 4;
        if ( v24 == v49 )
          goto LABEL_90;
      }
    }
    v51 = *(_QWORD **)(a1 + 64);
    v35 = (__int64)&v51[*(unsigned int *)(a1 + 76)];
    if ( v51 != (_QWORD *)v35 )
    {
      v52 = *(_QWORD **)(a1 + 64);
      do
      {
        if ( v50 == *v52 )
          goto LABEL_49;
        ++v52;
      }
      while ( (_QWORD *)v35 != v52 );
      v56 = v24[1];
      v34 = (__int64)(v24 + 1);
      if ( v46 == v56 )
      {
        ++v24;
        goto LABEL_49;
      }
LABEL_67:
      v57 = v51;
      do
      {
        if ( v56 == *v51 )
          goto LABEL_70;
        ++v51;
      }
      while ( (_QWORD *)v35 != v51 );
      v59 = v24[2];
      v34 = (__int64)(v24 + 2);
      if ( v46 == v59 )
        goto LABEL_70;
LABEL_75:
      v60 = v57;
      do
      {
        if ( v59 == *v57 )
          goto LABEL_70;
        ++v57;
      }
      while ( (_QWORD *)v35 != v57 );
      v62 = v24[3];
      v34 = (__int64)(v24 + 3);
      if ( v46 == v62 )
        goto LABEL_70;
      goto LABEL_83;
    }
    v34 = (__int64)(v24 + 1);
    if ( v46 == v24[1] )
      goto LABEL_70;
    goto LABEL_102;
  }
  v11 = a4;
  v12 = v85;
  v13 = *(_QWORD *)(a4 + 8);
  v14 = 0;
  v15 = *(_DWORD *)(v11 + 24);
  v16 = v15 - 1;
  do
  {
    v17 = *v12;
    if ( v15 )
    {
      v18 = v16 & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
      v19 = (__int64 *)(v13 + 16LL * v18);
      v20 = *v19;
      if ( v17 == *v19 )
      {
LABEL_8:
        v21 = (_QWORD *)v19[1];
        if ( v21 )
        {
          if ( !v14 )
            goto LABEL_4;
          if ( v14 != v21 )
          {
            v22 = (_QWORD *)v19[1];
            while ( 1 )
            {
              v22 = (_QWORD *)*v22;
              if ( v14 == v22 )
                break;
              if ( !v22 )
                goto LABEL_5;
            }
LABEL_4:
            v14 = v21;
          }
        }
      }
      else
      {
        v64 = 1;
        while ( v20 != -4096 )
        {
          v18 = v16 & (v64 + v18);
          v82 = v64 + 1;
          v19 = (__int64 *)(v13 + 16LL * v18);
          v20 = *v19;
          if ( v17 == *v19 )
            goto LABEL_8;
          v64 = v82;
        }
      }
    }
LABEL_5:
    ++v12;
  }
  while ( v12 != v10 );
  v79 = v14;
  if ( v14 != (_QWORD *)v6 )
  {
    v83 = a2;
    if ( v14 )
    {
      if ( v15 )
      {
        v37 = 1;
        v38 = 0;
        v39 = v16 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v40 = (char *)(v13 + 16LL * v39);
        v41 = *(_QWORD *)v40;
        if ( a2 == *(_QWORD *)v40 )
        {
LABEL_36:
          v42 = v40 + 8;
          goto LABEL_37;
        }
        while ( v41 != -4096 )
        {
          if ( v41 == -8192 && !v38 )
            v38 = v40;
          v39 = v16 & (v37 + v39);
          v40 = (char *)(v13 + 16LL * v39);
          v41 = *(_QWORD *)v40;
          if ( a2 == *(_QWORD *)v40 )
            goto LABEL_36;
          ++v37;
        }
        if ( v38 )
          v40 = v38;
        ++*(_QWORD *)a4;
        v72 = *(_DWORD *)(a4 + 16);
        v84 = v40;
        v71 = v72 + 1;
        if ( 4 * (v72 + 1) < 3 * v15 )
        {
          v70 = a2;
          if ( v15 - *(_DWORD *)(a4 + 20) - v71 <= v15 >> 3 )
          {
            v69 = a4;
            sub_D4F150(a4, v15);
LABEL_126:
            sub_D4C730(v69, &v83, &v84);
            v70 = v83;
            v71 = *(_DWORD *)(v69 + 16) + 1;
            v40 = v84;
          }
          *(_DWORD *)(a4 + 16) = v71;
          if ( *(_QWORD *)v40 != -4096 )
            --*(_DWORD *)(a4 + 20);
          *(_QWORD *)v40 = v70;
          v42 = v40 + 8;
          *v42 = 0;
LABEL_37:
          *v42 = v79;
          goto LABEL_38;
        }
      }
      else
      {
        v84 = 0;
        ++*(_QWORD *)a4;
      }
      v69 = a4;
      sub_D4F150(a4, 2 * v15);
      goto LABEL_126;
    }
    goto LABEL_121;
  }
  if ( v85 != (__int64 *)v87 )
    _libc_free((unsigned __int64)v85);
}
