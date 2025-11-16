// Function: sub_23C18D0
// Address: 0x23c18d0
//
__int64 __fastcall sub_23C18D0(__int64 a1, const void *a2, size_t a3, __int64 *a4)
{
  __int64 v5; // r12
  __int64 v6; // rax
  __int64 v7; // r13
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r12
  int v11; // r9d
  int v12; // esi
  unsigned int i; // eax
  __int64 v14; // rdi
  unsigned int v15; // eax
  __int64 v16; // rdi
  __int64 v17; // r12
  const void **v18; // r14
  void *(*v19)(); // rax
  void *v20; // rax
  __int64 v21; // r12
  __int64 result; // rax
  __int64 v23; // rdx
  __int64 v24; // rsi
  int v25; // r8d
  __int64 v26; // rcx
  unsigned int v27; // eax
  __int64 v28; // rbx
  __int64 v29; // rax
  __int64 v30; // rbx
  int v31; // r9d
  unsigned int j; // eax
  __int64 v33; // rdi
  unsigned int v34; // eax
  __int64 v35; // rbx
  __int64 v36; // rsi
  size_t v37; // rdx
  __int64 v38; // rax
  __int64 v39; // rax
  __int64 v40; // rax
  __int64 v41; // rax
  __int64 v42; // rdi
  __int64 v43; // rax
  __int64 v44; // rax
  __int64 *v45; // rbx
  __int64 *v46; // r12
  __int64 v47; // rax
  _QWORD *v48; // rdi
  const void **v49; // rsi
  const void *v50; // rax
  __int64 v51; // rcx
  const void **v52; // r13
  __int64 v53; // rbx
  __int64 v54; // r15
  int v55; // r12d
  __int64 v56; // r8
  unsigned int v57; // ecx
  const char *v58; // rax
  __int64 v59; // rdx
  unsigned __int64 v62; // [rsp+10h] [rbp-170h]
  unsigned __int64 v63; // [rsp+18h] [rbp-168h]
  size_t v64; // [rsp+30h] [rbp-150h]
  const char *v65; // [rsp+38h] [rbp-148h]
  int v66; // [rsp+54h] [rbp-12Ch]
  __int64 *v67; // [rsp+58h] [rbp-128h]
  __int64 v68; // [rsp+60h] [rbp-120h]
  const void **v69; // [rsp+68h] [rbp-118h]
  __int64 v70; // [rsp+70h] [rbp-110h]
  __int64 v71; // [rsp+80h] [rbp-100h]
  __int64 *v72; // [rsp+88h] [rbp-F8h]
  const void *v73; // [rsp+90h] [rbp-F0h] BYREF
  size_t v74; // [rsp+98h] [rbp-E8h]
  __int64 *v75; // [rsp+A0h] [rbp-E0h] BYREF
  int v76; // [rsp+A8h] [rbp-D8h]
  char v77; // [rsp+B0h] [rbp-D0h] BYREF
  _QWORD v78[4]; // [rsp+C0h] [rbp-C0h] BYREF
  __int16 v79; // [rsp+E0h] [rbp-A0h]
  const char *v80; // [rsp+F0h] [rbp-90h] BYREF
  __int64 v81; // [rsp+F8h] [rbp-88h]
  void ***v82; // [rsp+100h] [rbp-80h]
  __int64 v83; // [rsp+108h] [rbp-78h]
  char v84; // [rsp+110h] [rbp-70h]
  void *v85; // [rsp+118h] [rbp-68h] BYREF
  const void **v86; // [rsp+120h] [rbp-60h]
  void **v87; // [rsp+128h] [rbp-58h] BYREF
  const char *v88; // [rsp+130h] [rbp-50h]
  __int64 v89; // [rsp+138h] [rbp-48h]
  _QWORD v90[8]; // [rsp+140h] [rbp-40h] BYREF

  v5 = *(_QWORD *)(a1 + 8);
  v73 = a2;
  v74 = a3;
  sub_23B2720(&v80, a4);
  v6 = sub_23B66B0((__int64 *)&v80, 1);
  v7 = *(_QWORD *)(sub_BC0510(v5, &unk_4F82418, v6) + 8);
  sub_23B42E0((__int64 *)&v80);
  sub_23B2720(&v80, a4);
  sub_23B4300((__int64)&v75, (__int64 *)&v80);
  sub_23B42E0((__int64 *)&v80);
  v67 = &v75[v76];
  if ( v75 != v67 )
  {
    v72 = v75;
    while ( 1 )
    {
      v8 = *(unsigned int *)(v7 + 88);
      v9 = *(_QWORD *)(v7 + 72);
      v10 = *v72;
      if ( !(_DWORD)v8 )
        goto LABEL_17;
      v11 = 1;
      v12 = v8 - 1;
      v63 = (unsigned __int64)(((unsigned int)&unk_4FDE330 >> 9) ^ ((unsigned int)&unk_4FDE330 >> 4)) << 32;
      for ( i = (v8 - 1)
              & (((0xBF58476D1CE4E5B9LL * (v63 | ((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4))) >> 31)
               ^ (484763065 * (v63 | ((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4)))); ; i = v12 & v15 )
      {
        v14 = v9 + 24LL * i;
        if ( *(_UNKNOWN **)v14 == &unk_4FDE330 && v10 == *(_QWORD *)(v14 + 8) )
          break;
        if ( *(_QWORD *)v14 == -4096 && *(_QWORD *)(v14 + 8) == -4096 )
          goto LABEL_41;
        v15 = v11 + i;
        ++v11;
      }
      if ( v14 == v9 + 24LL * (unsigned int)v8 )
        goto LABEL_41;
      v29 = *(_QWORD *)(*(_QWORD *)(v14 + 16) + 24LL);
      if ( v29 )
        break;
LABEL_43:
      v31 = 1;
      v62 = (unsigned __int64)(((unsigned int)&unk_4FDE338 >> 9) ^ ((unsigned int)&unk_4FDE338 >> 4)) << 32;
      for ( j = v12
              & (((0xBF58476D1CE4E5B9LL * (v62 | ((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4))) >> 31)
               ^ (484763065 * (v62 | ((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4)))); ; j = v12 & v34 )
      {
        v33 = v9 + 24LL * j;
        if ( *(_UNKNOWN **)v33 == &unk_4FDE338 && v10 == *(_QWORD *)(v33 + 8) )
          break;
        if ( *(_QWORD *)v33 == -4096 && *(_QWORD *)(v33 + 8) == -4096 )
          goto LABEL_17;
        v34 = v31 + j;
        ++v31;
      }
      if ( v33 != v9 + 24 * v8 )
      {
        v35 = *(_QWORD *)(*(_QWORD *)(v33 + 16) + 24LL);
        if ( v35 )
        {
          v36 = v10;
          v68 = v35 + 8;
          sub_23C0920((__int64)&v80, v10, 0);
          v16 = v10;
          v65 = sub_BD5D20(v10);
          v64 = v37;
          if ( v84 )
          {
            v16 = (__int64)&v80;
            if ( sub_23AED50((__int64)&v80) )
              goto LABEL_53;
            if ( !*(_BYTE *)(v35 + 40) )
              goto LABEL_52;
          }
          else if ( !*(_BYTE *)(v35 + 40) )
          {
LABEL_52:
            if ( (_DWORD)v87 != *(_DWORD *)(v35 + 64) )
              goto LABEL_53;
            v16 = (unsigned int)v88;
            v17 = (__int64)v86;
            v66 = (int)v88;
            v18 = &v86[5 * (unsigned int)v88];
            if ( (_DWORD)v87 && v18 != v86 )
            {
              v49 = v86;
              while ( 1 )
              {
                v50 = *v49;
                if ( *v49 != (const void *)-8192LL && v50 != (const void *)-4096LL )
                  break;
                v49 += 5;
                if ( v18 == v49 )
                  goto LABEL_9;
              }
              if ( v18 == v49 )
                goto LABEL_83;
              v51 = *(_QWORD *)(v35 + 56);
              v70 = v7;
              v52 = v49;
              v69 = v86;
              v53 = *(unsigned int *)(v35 + 72);
              v36 = 5 * v53;
              v54 = v51;
              v71 = v51 + 40 * v53;
              v55 = v53 - 1;
              while ( 1 )
              {
                if ( !(_DWORD)v53 )
                  goto LABEL_53;
                v36 = v55 & (((unsigned int)v50 >> 9) ^ ((unsigned int)v50 >> 4));
                v16 = v54 + 40 * v36;
                v56 = *(_QWORD *)v16;
                if ( v50 != *(const void **)v16 )
                  break;
LABEL_76:
                if ( v71 == v16 )
                  goto LABEL_53;
                v16 += 8;
                v36 = (__int64)(v52 + 1);
                if ( !(unsigned __int8)sub_23B56C0(v16, (__int64)(v52 + 1)) )
                  goto LABEL_53;
                do
                {
                  v52 += 5;
                  if ( v52 == v18 )
                    goto LABEL_82;
                  v50 = *v52;
                }
                while ( *v52 == (const void *)-8192LL || v50 == (const void *)-4096LL );
                if ( v52 == v18 )
                {
LABEL_82:
                  v7 = v70;
                  v17 = (__int64)v69;
LABEL_83:
                  if ( !v66 )
                  {
LABEL_84:
                    v16 = 0;
                    goto LABEL_16;
                  }
                  do
                  {
LABEL_11:
                    if ( *(_QWORD *)v17 != -4096 && *(_QWORD *)v17 != -8192 )
                      sub_C7D6A0(*(_QWORD *)(v17 + 16), 16LL * *(unsigned int *)(v17 + 32), 8);
                    v17 += 40;
                  }
                  while ( (const void **)v17 != v18 );
                  v17 = (__int64)v86;
                  v16 = (unsigned int)v88;
                  goto LABEL_16;
                }
              }
              v16 = 1;
              while ( v56 != -4096 )
              {
                v57 = v16 + 1;
                v36 = v55 & (unsigned int)(v16 + v36);
                v16 = v54 + 40LL * (unsigned int)v36;
                v56 = *(_QWORD *)v16;
                if ( *(const void **)v16 == v50 )
                  goto LABEL_76;
                v16 = v57;
              }
LABEL_53:
              v38 = sub_C5F790(v16, v36);
              v39 = sub_904010(v38, "Error: ");
              v40 = sub_A51340(v39, v73, v74);
              v41 = sub_904010(v40, " does not invalidate CFG analyses but CFG changes detected in function @");
              v42 = sub_A51340(v41, v65, v64);
              sub_904010(v42, ":\n");
              v43 = sub_C5F790(v42, (__int64)":\n");
              sub_23B5800(v43, v68, (__int64)&v80);
              v78[0] = "CFG unexpectedly changed by ";
              v79 = 1283;
              v78[2] = v73;
              v78[3] = v74;
              sub_C64D30((__int64)v78, 1u);
            }
LABEL_9:
            if ( !(_DWORD)v88 )
              goto LABEL_84;
            if ( v18 != v86 )
              goto LABEL_11;
LABEL_16:
            sub_C7D6A0(v17, 40 * v16, 8);
            if ( v84 )
            {
              v44 = (unsigned int)v83;
              v84 = 0;
              if ( (_DWORD)v83 )
              {
                v45 = (__int64 *)v81;
                v46 = (__int64 *)(v81 + 40LL * (unsigned int)v83);
                do
                {
                  while ( 1 )
                  {
                    if ( *v45 <= 0x7FFFFFFFFFFFFFFDLL )
                    {
                      v45[1] = (__int64)&unk_49DB368;
                      v47 = v45[4];
                      if ( v47 != -4096 && v47 != 0 && v47 != -8192 )
                        break;
                    }
                    v45 += 5;
                    if ( v46 == v45 )
                      goto LABEL_65;
                  }
                  v48 = v45 + 2;
                  v45 += 5;
                  sub_BD60C0(v48);
                }
                while ( v46 != v45 );
LABEL_65:
                v44 = (unsigned int)v83;
              }
              sub_C7D6A0(v81, 40 * v44, 8);
            }
            goto LABEL_17;
          }
          v16 = v35 + 8;
          if ( sub_23AED50(v68) )
            goto LABEL_53;
          goto LABEL_52;
        }
      }
LABEL_17:
      if ( v67 == ++v72 )
      {
        v67 = v75;
        goto LABEL_19;
      }
    }
    v30 = *(_QWORD *)(v29 + 8);
    if ( v30 != sub_3148040(*v72, 0) )
    {
      v58 = sub_BD5D20(v10);
      v84 = 1;
      v80 = "Function @{0} changed by {1} without invalidating analyses";
      v82 = (void ***)v90;
      v88 = v58;
      v90[0] = &v87;
      v85 = &unk_49DB108;
      v86 = &v73;
      v81 = 58;
      v83 = 2;
      v87 = (void **)&unk_4A09A78;
      v89 = v59;
      v90[1] = &v85;
LABEL_96:
      v79 = 263;
      v78[0] = &v80;
      sub_C64D30((__int64)v78, 1u);
    }
    v9 = *(_QWORD *)(v7 + 72);
    v8 = *(unsigned int *)(v7 + 88);
LABEL_41:
    if ( !(_DWORD)v8 )
      goto LABEL_17;
    v12 = v8 - 1;
    goto LABEL_43;
  }
LABEL_19:
  if ( v67 != (__int64 *)&v77 )
    _libc_free((unsigned __int64)v67);
  sub_23B2720(&v80, a4);
  if ( !v80 )
    return sub_23B42E0((__int64 *)&v80);
  v19 = *(void *(**)())(*(_QWORD *)v80 + 24LL);
  v20 = v19 == sub_23AE340 ? &unk_4CDFBF8 : v19();
  if ( v20 != &unk_4C5D162 )
    return sub_23B42E0((__int64 *)&v80);
  v21 = *((_QWORD *)v80 + 1);
  result = sub_23B42E0((__int64 *)&v80);
  if ( v21 )
  {
    result = *(_QWORD *)(a1 + 8);
    v23 = *(unsigned int *)(result + 88);
    v24 = *(_QWORD *)(result + 72);
    if ( (_DWORD)v23 )
    {
      v25 = 1;
      for ( result = ((_DWORD)v23 - 1)
                   & ((unsigned int)((0xBF58476D1CE4E5B9LL
                                    * (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4)
                                     | ((unsigned __int64)(((unsigned int)&unk_4FDE328 >> 9)
                                                         ^ ((unsigned int)&unk_4FDE328 >> 4)) << 32))) >> 31)
                    ^ (484763065 * (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4))));
            ;
            result = ((_DWORD)v23 - 1) & v27 )
      {
        v26 = v24 + 24LL * (unsigned int)result;
        if ( *(_UNKNOWN **)v26 == &unk_4FDE328 && v21 == *(_QWORD *)(v26 + 8) )
          break;
        if ( *(_QWORD *)v26 == -4096 && *(_QWORD *)(v26 + 8) == -4096 )
          return result;
        v27 = v25 + result;
        ++v25;
      }
      result = v24 + 24 * v23;
      if ( v26 != result )
      {
        result = *(_QWORD *)(*(_QWORD *)(v26 + 16) + 24LL);
        if ( result )
        {
          v28 = *(_QWORD *)(result + 8);
          result = sub_3147DF0(v21, 0);
          if ( v28 != result )
          {
            v83 = 1;
            v80 = "Module changed by {0} without invalidating analyses";
            v82 = &v87;
            v81 = 51;
            v84 = 1;
            v85 = &unk_49DB108;
            v86 = &v73;
            v87 = &v85;
            goto LABEL_96;
          }
        }
      }
    }
  }
  return result;
}
