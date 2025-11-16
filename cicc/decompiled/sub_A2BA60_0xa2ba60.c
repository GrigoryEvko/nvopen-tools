// Function: sub_A2BA60
// Address: 0xa2ba60
//
_QWORD *__fastcall sub_A2BA60(_DWORD **a1, __int64 a2, __int64 a3)
{
  _QWORD *result; // rax
  _QWORD *v5; // rsi
  __int64 v6; // r12
  char **v7; // rsi
  unsigned __int64 v8; // rdx
  char *v9; // r9
  char *v10; // rbx
  char *v11; // r15
  __int64 v12; // r12
  bool v13; // zf
  int v14; // r13d
  __int64 v15; // rax
  unsigned int v16; // r8d
  __int64 v17; // rbx
  __int64 v18; // r9
  unsigned int v19; // r8d
  _QWORD *v20; // rax
  __int64 v21; // rdi
  char *v22; // rdi
  size_t v23; // rdx
  int v24; // r10d
  int v25; // r10d
  __int64 v26; // r9
  unsigned int v27; // edx
  int v28; // eax
  _QWORD *v29; // rcx
  int v30; // r8d
  _QWORD *v31; // rdi
  int v32; // eax
  __int64 v33; // rdx
  int v34; // eax
  _QWORD *v35; // rdx
  __int64 v36; // r15
  __int64 v37; // r8
  _QWORD *v38; // rax
  __int64 v39; // rdx
  __int64 v40; // rdx
  __int64 *v41; // rbx
  __int64 *v42; // r13
  __int64 v43; // r12
  __int64 v44; // rcx
  __int64 v45; // rdi
  __int64 *v46; // rdx
  __int64 *v47; // r12
  __int64 *v48; // rbx
  __int64 *v49; // rdi
  int v50; // eax
  int v51; // r10d
  int v52; // r10d
  __int64 v53; // r9
  int v54; // r8d
  unsigned int v55; // edx
  __int64 v56; // [rsp-128h] [rbp-128h]
  _QWORD *v58; // [rsp-118h] [rbp-118h]
  unsigned int v59; // [rsp-110h] [rbp-110h]
  __int64 v60; // [rsp-110h] [rbp-110h]
  _QWORD *v61; // [rsp-108h] [rbp-108h]
  char **v62; // [rsp-100h] [rbp-100h]
  __int64 v63; // [rsp-F0h] [rbp-F0h]
  unsigned int v64; // [rsp-F0h] [rbp-F0h]
  int v65; // [rsp-F0h] [rbp-F0h]
  __int64 v66; // [rsp-F0h] [rbp-F0h]
  int v67; // [rsp-F0h] [rbp-F0h]
  unsigned __int64 v68; // [rsp-F0h] [rbp-F0h]
  char **v69; // [rsp-E0h] [rbp-E0h]
  int v70; // [rsp-D0h] [rbp-D0h] BYREF
  _DWORD *v71; // [rsp-C8h] [rbp-C8h] BYREF
  __int64 v72; // [rsp-C0h] [rbp-C0h]
  _BYTE v73[48]; // [rsp-B8h] [rbp-B8h] BYREF
  __int64 v74; // [rsp-88h] [rbp-88h] BYREF
  char *v75; // [rsp-80h] [rbp-80h] BYREF
  __int64 v76; // [rsp-78h] [rbp-78h]
  _BYTE v77[112]; // [rsp-70h] [rbp-70h] BYREF

  result = a1[13];
  if ( result )
  {
    v5 = (_QWORD *)*result;
    result = (_QWORD *)result[1];
    v61 = v5;
    v58 = result;
    if ( v5 != result )
    {
      while ( 1 )
      {
        v6 = a3;
        v7 = (char **)v61[8];
        v69 = v7;
        v62 = (char **)v61[9];
        if ( v7 != v62 )
          break;
LABEL_21:
        v61 += 14;
        result = v61;
        if ( v61 == v58 )
          return result;
      }
      while ( 1 )
      {
        v71 = v73;
        v72 = 0xC00000000LL;
        v8 = *((unsigned int *)v69 + 4);
        if ( (unsigned int)v8 > 0xC )
        {
          v7 = (char **)v73;
          a1 = &v71;
          sub_C8D5F0(&v71, v73, v8, 4);
          v8 = *((unsigned int *)v69 + 4);
        }
        v9 = v69[1];
        v10 = &v9[4 * v8];
        if ( v10 == v9 )
        {
          v16 = v72;
        }
        else
        {
          v63 = v6;
          v11 = v69[1];
          v12 = a2;
          do
          {
            v13 = *(_QWORD *)(v12 + 16) == 0;
            LODWORD(v74) = *(_DWORD *)v11;
            if ( v13 )
              sub_4263D6(a1, v7, v8);
            v7 = (char **)&v74;
            a1 = (_DWORD **)v12;
            v14 = (*(__int64 (__fastcall **)(__int64, __int64 *))(v12 + 24))(v12, &v74);
            v15 = (unsigned int)v72;
            if ( (unsigned __int64)(unsigned int)v72 + 1 > HIDWORD(v72) )
            {
              v7 = (char **)v73;
              a1 = &v71;
              sub_C8D5F0(&v71, v73, (unsigned int)v72 + 1LL, 4);
              v15 = (unsigned int)v72;
            }
            v8 = (unsigned __int64)v71;
            v11 += 4;
            v71[v15] = v14;
            v16 = v72 + 1;
            LODWORD(v72) = v72 + 1;
          }
          while ( v10 != v11 );
          a2 = v12;
          v6 = v63;
        }
        v17 = *(unsigned int *)(v6 + 40);
        v75 = v77;
        v74 = v17;
        v76 = 0xC00000000LL;
        if ( v16 )
        {
          v22 = v77;
          v23 = 4LL * v16;
          if ( v16 <= 0xC
            || (v59 = v16, sub_C8D5F0(&v75, v77, v16, 4), v22 = v75, v16 = v59, (v23 = 4LL * (unsigned int)v72) != 0) )
          {
            v64 = v16;
            memcpy(v22, v71, v23);
            v16 = v64;
          }
          v7 = (char **)*(unsigned int *)(v6 + 24);
          v17 = v74;
          LODWORD(v76) = v16;
          if ( !(_DWORD)v7 )
          {
LABEL_26:
            ++*(_QWORD *)v6;
            goto LABEL_27;
          }
        }
        else
        {
          v7 = (char **)*(unsigned int *)(v6 + 24);
          if ( !(_DWORD)v7 )
            goto LABEL_26;
        }
        v18 = *(_QWORD *)(v6 + 8);
        v19 = (((0xBF58476D1CE4E5B9LL * v17) >> 31) ^ (484763065 * v17)) & ((_DWORD)v7 - 1);
        v20 = (_QWORD *)(v18 + 16LL * v19);
        v21 = *v20;
        if ( *v20 != v17 )
          break;
LABEL_16:
        if ( v75 != v77 )
          _libc_free(v75, v7);
        a1 = (_DWORD **)v71;
        if ( v71 != (_DWORD *)v73 )
          _libc_free(v71, v7);
        v69 += 9;
        if ( v62 == v69 )
          goto LABEL_21;
      }
      v65 = 1;
      v29 = 0;
      while ( v21 != -1 )
      {
        if ( v29 || v21 != -2 )
          v20 = v29;
        v19 = ((_DWORD)v7 - 1) & (v65 + v19);
        v21 = *(_QWORD *)(v18 + 16LL * v19);
        if ( v17 == v21 )
          goto LABEL_16;
        ++v65;
        v29 = v20;
        v20 = (_QWORD *)(v18 + 16LL * v19);
      }
      if ( !v29 )
        v29 = v20;
      v32 = *(_DWORD *)(v6 + 16);
      ++*(_QWORD *)v6;
      v28 = v32 + 1;
      if ( 4 * v28 >= (unsigned int)(3 * (_DWORD)v7) )
      {
LABEL_27:
        sub_9E25D0(v6, 2 * (_DWORD)v7);
        v24 = *(_DWORD *)(v6 + 24);
        if ( !v24 )
          goto LABEL_90;
        v25 = v24 - 1;
        v26 = *(_QWORD *)(v6 + 8);
        v27 = v25 & (((0xBF58476D1CE4E5B9LL * v17) >> 31) ^ (484763065 * v17));
        v28 = *(_DWORD *)(v6 + 16) + 1;
        v29 = (_QWORD *)(v26 + 16LL * v27);
        v7 = (char **)*v29;
        if ( v17 == *v29 )
          goto LABEL_45;
        v30 = 1;
        v31 = 0;
        while ( v7 != (char **)-1LL )
        {
          if ( !v31 && v7 == (char **)-2LL )
            v31 = v29;
          v27 = v25 & (v30 + v27);
          v29 = (_QWORD *)(v26 + 16LL * v27);
          v7 = (char **)*v29;
          if ( v17 == *v29 )
            goto LABEL_45;
          ++v30;
        }
      }
      else
      {
        if ( (int)v7 - *(_DWORD *)(v6 + 20) - v28 > (unsigned int)v7 >> 3 )
          goto LABEL_45;
        v68 = ((0xBF58476D1CE4E5B9LL * v17) >> 31) ^ (0xBF58476D1CE4E5B9LL * v17);
        sub_9E25D0(v6, (int)v7);
        v51 = *(_DWORD *)(v6 + 24);
        if ( !v51 )
        {
LABEL_90:
          ++*(_DWORD *)(v6 + 16);
          BUG();
        }
        v52 = v51 - 1;
        v53 = *(_QWORD *)(v6 + 8);
        v31 = 0;
        v54 = 1;
        v28 = *(_DWORD *)(v6 + 16) + 1;
        v55 = v52 & v68;
        v29 = (_QWORD *)(v53 + 16LL * (v52 & (unsigned int)v68));
        v7 = (char **)*v29;
        if ( v17 == *v29 )
          goto LABEL_45;
        while ( v7 != (char **)-1LL )
        {
          if ( v7 == (char **)-2LL && !v31 )
            v31 = v29;
          v55 = v52 & (v54 + v55);
          v29 = (_QWORD *)(v53 + 16LL * v55);
          v7 = (char **)*v29;
          if ( v17 == *v29 )
            goto LABEL_45;
          ++v54;
        }
      }
      if ( v31 )
        v29 = v31;
LABEL_45:
      *(_DWORD *)(v6 + 16) = v28;
      if ( *v29 != -1 )
        --*(_DWORD *)(v6 + 20);
      *((_DWORD *)v29 + 2) = 0;
      *v29 = v17;
      *((_DWORD *)v29 + 2) = *(_DWORD *)(v6 + 40);
      v33 = *(unsigned int *)(v6 + 40);
      v34 = v33;
      if ( *(_DWORD *)(v6 + 44) <= (unsigned int)v33 )
      {
        v7 = (char **)(v6 + 48);
        v66 = v6 + 48;
        v36 = sub_C8D7D0(v6 + 32, v6 + 48, 0, 72, &v70);
        v37 = 72LL * *(unsigned int *)(v6 + 40);
        v38 = (_QWORD *)(v37 + v36);
        if ( v37 + v36 )
        {
          v7 = (char **)0xC00000000LL;
          v39 = v74;
          v38[2] = 0xC00000000LL;
          *v38 = v39;
          v38[1] = v38 + 3;
          if ( (_DWORD)v76 )
          {
            v7 = &v75;
            sub_A15BD0((__int64)(v38 + 1), &v75);
          }
          v37 = 72LL * *(unsigned int *)(v6 + 40);
        }
        v40 = *(_QWORD *)(v6 + 32);
        v41 = (__int64 *)(v40 + v37);
        if ( v40 != v40 + v37 )
        {
          v60 = a2;
          v42 = *(__int64 **)(v6 + 32);
          v56 = v6;
          v43 = v36;
          do
          {
            while ( 1 )
            {
              if ( v43 )
              {
                v44 = *v42;
                *(_DWORD *)(v43 + 16) = 0;
                *(_DWORD *)(v43 + 20) = 12;
                *(_QWORD *)v43 = v44;
                *(_QWORD *)(v43 + 8) = v43 + 24;
                if ( *((_DWORD *)v42 + 4) )
                  break;
              }
              v42 += 9;
              v43 += 72;
              if ( v41 == v42 )
                goto LABEL_63;
            }
            v7 = (char **)(v42 + 1);
            v45 = v43 + 8;
            v42 += 9;
            v43 += 72;
            sub_A15BD0(v45, v7);
          }
          while ( v41 != v42 );
LABEL_63:
          v6 = v56;
          a2 = v60;
          v46 = *(__int64 **)(v56 + 32);
          v41 = &v46[9 * *(unsigned int *)(v56 + 40)];
          if ( v46 != v41 )
          {
            v47 = &v46[9 * *(unsigned int *)(v56 + 40)];
            v48 = *(__int64 **)(v56 + 32);
            do
            {
              v47 -= 9;
              v49 = (__int64 *)v47[1];
              if ( v49 != v47 + 3 )
                _libc_free(v49, v7);
            }
            while ( v48 != v47 );
            v6 = v56;
            v41 = *(__int64 **)(v56 + 32);
          }
        }
        v50 = v70;
        if ( v41 != (__int64 *)v66 )
        {
          v67 = v70;
          _libc_free(v41, v7);
          v50 = v67;
        }
        ++*(_DWORD *)(v6 + 40);
        *(_QWORD *)(v6 + 32) = v36;
        *(_DWORD *)(v6 + 44) = v50;
      }
      else
      {
        v35 = (_QWORD *)(*(_QWORD *)(v6 + 32) + 72 * v33);
        if ( v35 )
        {
          *v35 = v74;
          v35[1] = v35 + 3;
          v35[2] = 0xC00000000LL;
          if ( (_DWORD)v76 )
          {
            v7 = &v75;
            sub_A15BD0((__int64)(v35 + 1), &v75);
          }
          v34 = *(_DWORD *)(v6 + 40);
        }
        *(_DWORD *)(v6 + 40) = v34 + 1;
      }
      goto LABEL_16;
    }
  }
  return result;
}
