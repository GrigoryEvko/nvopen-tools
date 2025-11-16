// Function: sub_15127D0
// Address: 0x15127d0
//
bool __fastcall sub_15127D0(__int64 a1, int a2, _DWORD *a3)
{
  __int64 v4; // rbx
  int v5; // r13d
  unsigned int v6; // eax
  __int64 v7; // rcx
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rdx
  _QWORD *v11; // rax
  char *v12; // rcx
  char *v13; // r13
  __int64 v14; // rsi
  char **v15; // rax
  char *v16; // rdx
  char *v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rdi
  char *v20; // rdx
  char *v21; // r15
  char *v22; // r8
  __int64 v23; // rdi
  _QWORD *v24; // rax
  char *v25; // rcx
  __int64 v26; // rdx
  unsigned int v27; // r8d
  __int64 v28; // r9
  unsigned __int64 v29; // rdi
  unsigned __int64 v30; // rax
  unsigned int v31; // r12d
  unsigned __int64 *v32; // r10
  unsigned __int64 v33; // rsi
  unsigned int v34; // r13d
  unsigned int v35; // edi
  unsigned int v36; // r13d
  char v37; // r12
  __int64 v38; // r14
  unsigned __int64 v39; // r11
  unsigned __int64 v40; // r8
  unsigned int v41; // r15d
  unsigned __int64 *v42; // r10
  unsigned __int64 v43; // rsi
  unsigned int v44; // r8d
  unsigned __int64 v45; // r14
  unsigned __int64 v46; // rax
  unsigned int v47; // eax
  __int64 v48; // r11
  unsigned __int64 v49; // rdi
  __int64 v50; // rax
  __int64 v51; // rdx
  char v52; // cl
  unsigned int v53; // r8d
  __int64 v54; // rax
  __int64 v55; // rdx
  char v56; // cl
  unsigned __int64 v57; // r9
  unsigned __int64 v58; // rax
  unsigned __int64 v59; // rdx
  unsigned __int64 *v60; // r8
  unsigned __int64 v61; // rdx
  int v62; // r10d
  int v63; // ecx
  bool result; // al
  unsigned __int64 v65; // rax
  unsigned __int64 v66; // rdx
  unsigned __int64 v67; // rax
  unsigned __int64 v68; // rax
  unsigned __int64 v69; // rax
  unsigned int v70; // edx
  __int64 v71; // r9
  unsigned int v72; // r10d
  unsigned __int64 v73; // rdi
  __int64 v74; // rax
  __int64 v75; // rsi
  char v76; // cl
  unsigned __int64 v77; // rdi
  unsigned __int64 v78; // rax
  bool v79; // cf
  unsigned __int64 v80; // rdi
  _QWORD *v81; // r14
  _QWORD *v82; // rax
  _QWORD *v83; // rdx
  __int64 v84; // rsi
  __int64 v85; // rsi
  char *v86; // rcx
  _QWORD *v87; // rax
  __int64 v88; // rsi
  char *v89; // r8
  _QWORD *v90; // r12
  __int64 v91; // rdx
  __int64 v92; // rdx
  __int64 v93; // rbx
  char *v94; // r15
  volatile signed __int32 *v95; // r13
  signed __int32 v96; // edx
  signed __int32 v97; // edx
  __int64 v98; // r12
  __int64 v99; // rax
  __int64 v100; // [rsp+8h] [rbp-48h]
  char *v101; // [rsp+8h] [rbp-48h]
  __int64 v102; // [rsp+10h] [rbp-40h]

  v4 = a1;
  v5 = *(_DWORD *)(a1 + 36);
  v6 = *(_DWORD *)(a1 + 72);
  if ( v6 >= *(_DWORD *)(a1 + 76) )
  {
    sub_14F2B60(a1 + 64, 0);
    v6 = *(_DWORD *)(a1 + 72);
  }
  v7 = *(_QWORD *)(a1 + 64);
  v8 = v7 + 32LL * v6;
  if ( v8 )
  {
    *(_DWORD *)v8 = v5;
    *(_QWORD *)(v8 + 8) = 0;
    *(_QWORD *)(v8 + 16) = 0;
    *(_QWORD *)(v8 + 24) = 0;
    v7 = *(_QWORD *)(a1 + 64);
    v6 = *(_DWORD *)(a1 + 72);
  }
  v9 = v6 + 1;
  v10 = *(_QWORD *)(a1 + 40);
  *(_DWORD *)(a1 + 72) = v9;
  v11 = (_QWORD *)(v7 + 32 * v9 - 32);
  v12 = (char *)v11[1];
  v13 = (char *)v11[2];
  v11[1] = v10;
  v14 = v11[3];
  v11[2] = *(_QWORD *)(a1 + 48);
  v11[3] = *(_QWORD *)(a1 + 56);
  v15 = *(char ***)(a1 + 336);
  *(_QWORD *)(a1 + 40) = v12;
  *(_QWORD *)(a1 + 48) = v13;
  *(_QWORD *)(a1 + 56) = v14;
  if ( v15 )
  {
    v16 = v15[1];
    v17 = *v15;
    if ( v16 != v17 && a2 == *((_DWORD *)v16 - 22) )
    {
      v20 = v16 - 88;
    }
    else
    {
      v18 = 0x2E8BA2E8BA2E8BA3LL * ((v16 - v17) >> 3);
      if ( !(_DWORD)v18 )
        goto LABEL_21;
      v19 = (__int64)&v17[88 * (unsigned int)(v18 - 1) + 88];
      while ( 1 )
      {
        v20 = v17;
        if ( a2 == *(_DWORD *)v17 )
          break;
        v17 += 88;
        if ( v17 == (char *)v19 )
          goto LABEL_21;
      }
    }
    v21 = (char *)*((_QWORD *)v20 + 2);
    v22 = (char *)*((_QWORD *)v20 + 1);
    if ( v22 != v21 )
    {
      v23 = v21 - v22;
      if ( v14 - (__int64)v13 >= (unsigned __int64)(v21 - v22) )
      {
        v24 = (_QWORD *)*((_QWORD *)v20 + 1);
        v25 = &v13[v23];
        do
        {
          if ( v13 )
          {
            *(_QWORD *)v13 = *v24;
            v26 = v24[1];
            *((_QWORD *)v13 + 1) = v26;
            if ( v26 )
            {
              if ( &_pthread_key_create )
                _InterlockedAdd((volatile signed __int32 *)(v26 + 8), 1u);
              else
                ++*(_DWORD *)(v26 + 8);
            }
          }
          v13 += 16;
          v24 += 2;
        }
        while ( v13 != v25 );
        *(_QWORD *)(v4 + 48) += v23;
        goto LABEL_21;
      }
      v77 = v23 >> 4;
      v78 = (v13 - v12) >> 4;
      if ( v77 > 0x7FFFFFFFFFFFFFFLL - v78 )
        sub_4262D8((__int64)"vector::_M_range_insert");
      if ( v77 < v78 )
        v77 = (v13 - v12) >> 4;
      v79 = __CFADD__(v78, v77);
      v80 = v78 + v77;
      if ( v79 )
      {
        v98 = 0x7FFFFFFFFFFFFFF0LL;
      }
      else
      {
        if ( !v80 )
        {
          v102 = 0;
          v81 = 0;
LABEL_77:
          if ( v13 == v12 )
          {
            v83 = v81;
          }
          else
          {
            v82 = v81;
            v83 = (_QWORD *)((char *)v81 + v13 - v12);
            do
            {
              if ( v82 )
              {
                v84 = *(_QWORD *)v12;
                v82[1] = 0;
                *v82 = v84;
                v85 = *((_QWORD *)v12 + 1);
                *((_QWORD *)v12 + 1) = 0;
                v82[1] = v85;
                *(_QWORD *)v12 = 0;
              }
              v82 += 2;
              v12 += 16;
            }
            while ( v83 != v82 );
          }
          v86 = v22;
          v87 = (_QWORD *)((char *)v83 + v21 - v22);
          do
          {
            if ( v83 )
            {
              *v83 = *(_QWORD *)v86;
              v88 = *((_QWORD *)v86 + 1);
              v83[1] = v88;
              if ( v88 )
              {
                if ( &_pthread_key_create )
                  _InterlockedAdd((volatile signed __int32 *)(v88 + 8), 1u);
                else
                  ++*(_DWORD *)(v88 + 8);
              }
            }
            v83 += 2;
            v86 += 16;
          }
          while ( v87 != v83 );
          v89 = *(char **)(v4 + 48);
          if ( v13 == v89 )
          {
            v90 = v87;
          }
          else
          {
            v90 = (_QWORD *)((char *)v87 + v89 - v13);
            do
            {
              if ( v87 )
              {
                v91 = *(_QWORD *)v13;
                v87[1] = 0;
                *v87 = v91;
                v92 = *((_QWORD *)v13 + 1);
                *((_QWORD *)v13 + 1) = 0;
                v87[1] = v92;
                *(_QWORD *)v13 = 0;
              }
              v87 += 2;
              v13 += 16;
            }
            while ( v90 != v87 );
            v89 = *(char **)(v4 + 48);
          }
          if ( *(char **)(v4 + 40) != v89 )
          {
            v100 = v4;
            v93 = *(_QWORD *)(v4 + 40);
            v94 = v89;
            do
            {
              v95 = *(volatile signed __int32 **)(v93 + 8);
              if ( v95 )
              {
                if ( &_pthread_key_create )
                {
                  v96 = _InterlockedExchangeAdd(v95 + 2, 0xFFFFFFFF);
                }
                else
                {
                  v96 = *((_DWORD *)v95 + 2);
                  *((_DWORD *)v95 + 2) = v96 - 1;
                }
                if ( v96 == 1 )
                {
                  (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v95 + 16LL))(v95);
                  if ( &_pthread_key_create )
                  {
                    v97 = _InterlockedExchangeAdd(v95 + 3, 0xFFFFFFFF);
                  }
                  else
                  {
                    v97 = *((_DWORD *)v95 + 3);
                    *((_DWORD *)v95 + 3) = v97 - 1;
                  }
                  if ( v97 == 1 )
                    (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v95 + 24LL))(v95);
                }
              }
              v93 += 16;
            }
            while ( v94 != (char *)v93 );
            v4 = v100;
            v89 = *(char **)(v100 + 40);
          }
          if ( v89 )
            j_j___libc_free_0(v89, *(_QWORD *)(v4 + 56) - (_QWORD)v89);
          *(_QWORD *)(v4 + 40) = v81;
          *(_QWORD *)(v4 + 48) = v90;
          *(_QWORD *)(v4 + 56) = v102;
          goto LABEL_21;
        }
        if ( v80 > 0x7FFFFFFFFFFFFFFLL )
          v80 = 0x7FFFFFFFFFFFFFFLL;
        v98 = 16 * v80;
      }
      v101 = (char *)*((_QWORD *)v20 + 1);
      v99 = sub_22077B0(v98);
      v12 = *(char **)(v4 + 40);
      v22 = v101;
      v81 = (_QWORD *)v99;
      v102 = v98 + v99;
      goto LABEL_77;
    }
  }
LABEL_21:
  v27 = *(_DWORD *)(v4 + 32);
  if ( v27 > 3 )
  {
    v65 = *(_QWORD *)(v4 + 24);
    v35 = v27 - 4;
    *(_DWORD *)(v4 + 32) = v27 - 4;
    v66 = v65;
    v67 = v65 & 0xF;
    LOBYTE(v36) = v67;
    *(_QWORD *)(v4 + 24) = v66 >> 4;
    if ( v67 >> 3 )
      goto LABEL_28;
    *(_DWORD *)(v4 + 36) = v67;
    if ( v35 <= 0x1F )
      goto LABEL_51;
  }
  else
  {
    LODWORD(v28) = 0;
    if ( v27 )
      v28 = *(_QWORD *)(v4 + 24);
    v29 = *(_QWORD *)(v4 + 16);
    v30 = *(_QWORD *)(v4 + 8);
    v31 = 4 - v27;
    if ( v29 >= v30 )
      goto LABEL_113;
    v32 = (unsigned __int64 *)(v29 + *(_QWORD *)v4);
    if ( v30 < v29 + 8 )
    {
      v47 = v30 - v29;
      *(_QWORD *)(v4 + 24) = 0;
      v48 = v47;
      v34 = 8 * v47;
      v49 = v47 + v29;
      if ( !v47 )
      {
        *(_QWORD *)(v4 + 16) = v49;
LABEL_118:
        *(_DWORD *)(v4 + 32) = 0;
        goto LABEL_113;
      }
      v50 = 0;
      v33 = 0;
      do
      {
        v51 = *((unsigned __int8 *)v32 + v50);
        v52 = 8 * v50++;
        v33 |= v51 << v52;
        *(_QWORD *)(v4 + 24) = v33;
      }
      while ( v48 != v50 );
      *(_QWORD *)(v4 + 16) = v49;
      *(_DWORD *)(v4 + 32) = v34;
      if ( v31 > v34 )
        goto LABEL_113;
    }
    else
    {
      v33 = *v32;
      *(_QWORD *)(v4 + 16) = v29 + 8;
      v34 = 64;
    }
    v35 = v27 + v34 - 4;
    *(_DWORD *)(v4 + 32) = v35;
    *(_QWORD *)(v4 + 24) = v33 >> v31;
    v36 = v28 | (((0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v27 + 60)) & v33) << v27);
    if ( (v36 & 8) != 0 )
    {
LABEL_28:
      v36 &= 7u;
      v37 = 0;
      while ( 1 )
      {
        v37 += 3;
        if ( v35 <= 3 )
        {
          v38 = 0;
          if ( v35 )
            v38 = *(_QWORD *)(v4 + 24);
          v39 = *(_QWORD *)(v4 + 16);
          v40 = *(_QWORD *)(v4 + 8);
          v41 = 4 - v35;
          if ( v39 >= v40 )
            goto LABEL_113;
          v42 = (unsigned __int64 *)(v39 + *(_QWORD *)v4);
          if ( v40 < v39 + 8 )
          {
            *(_QWORD *)(v4 + 24) = 0;
            v53 = v40 - v39;
            if ( !v53 )
              goto LABEL_118;
            v54 = 0;
            v43 = 0;
            do
            {
              v55 = *((unsigned __int8 *)v42 + v54);
              v56 = 8 * v54++;
              v43 |= v55 << v56;
              *(_QWORD *)(v4 + 24) = v43;
            }
            while ( v53 != v54 );
            v57 = v39 + v53;
            v44 = 8 * v53;
            *(_QWORD *)(v4 + 16) = v57;
            *(_DWORD *)(v4 + 32) = v44;
            if ( v41 > v44 )
              goto LABEL_113;
          }
          else
          {
            v43 = *v42;
            *(_QWORD *)(v4 + 16) = v39 + 8;
            v44 = 64;
          }
          *(_QWORD *)(v4 + 24) = v43 >> v41;
          *(_DWORD *)(v4 + 32) = v35 + v44 - 4;
          v45 = (((0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v35 + 60)) & v43) << v35) | v38;
        }
        else
        {
          v46 = *(_QWORD *)(v4 + 24);
          *(_DWORD *)(v4 + 32) = v35 - 4;
          *(_QWORD *)(v4 + 24) = v46 >> 4;
          LOBYTE(v45) = v46 & 0xF;
        }
        v36 |= (v45 & 7) << v37;
        if ( (v45 & 8) == 0 )
          break;
        v35 = *(_DWORD *)(v4 + 32);
      }
    }
    *(_DWORD *)(v4 + 36) = v36;
    if ( v36 > 0x40 )
      return 1;
    v35 = *(_DWORD *)(v4 + 32);
    if ( v35 <= 0x1F )
    {
LABEL_51:
      v58 = *(_QWORD *)(v4 + 16);
      v59 = *(_QWORD *)(v4 + 8);
      *(_DWORD *)(v4 + 32) = 0;
      if ( v58 < v59 )
      {
        v60 = (unsigned __int64 *)(v58 + *(_QWORD *)v4);
        if ( v59 >= v58 + 8 )
        {
          v61 = *v60;
          *(_QWORD *)(v4 + 16) = v58 + 8;
          v62 = 32;
LABEL_54:
          v63 = v61;
          *(_DWORD *)(v4 + 32) = v62;
          *(_QWORD *)(v4 + 24) = HIDWORD(v61);
          goto LABEL_55;
        }
        v70 = v59 - v58;
        *(_QWORD *)(v4 + 24) = 0;
        v71 = v70;
        v72 = 8 * v70;
        v73 = v70 + v58;
        if ( v70 )
        {
          v74 = 0;
          v61 = 0;
          do
          {
            v75 = *((unsigned __int8 *)v60 + v74);
            v76 = 8 * v74++;
            v61 |= v75 << v76;
            *(_QWORD *)(v4 + 24) = v61;
          }
          while ( v71 != v74 );
          *(_QWORD *)(v4 + 16) = v73;
          *(_DWORD *)(v4 + 32) = v72;
          if ( v72 > 0x1F )
          {
            v62 = v72 - 32;
            goto LABEL_54;
          }
        }
        else
        {
          *(_QWORD *)(v4 + 16) = v73;
        }
      }
LABEL_113:
      sub_16BD130("Unexpected end of file", 1);
    }
  }
  v68 = *(_QWORD *)(v4 + 24);
  *(_DWORD *)(v4 + 32) = 0;
  v69 = v68 >> ((unsigned __int8)v35 - 32);
  v63 = v69;
  *(_QWORD *)(v4 + 24) = HIDWORD(v69);
LABEL_55:
  if ( a3 )
    *a3 = v63;
  if ( !*(_DWORD *)(v4 + 36) )
    return 1;
  result = 0;
  if ( !*(_DWORD *)(v4 + 32) )
    return *(_QWORD *)(v4 + 8) <= *(_QWORD *)(v4 + 16);
  return result;
}
