// Function: sub_1C12D00
// Address: 0x1c12d00
//
_QWORD *__fastcall sub_1C12D00(_QWORD *a1, __int64 a2, unsigned __int64 *a3, __int64 a4, __int64 a5, int a6)
{
  __int64 v7; // rcx
  _QWORD *v8; // rdx
  unsigned __int64 *v9; // rbx
  __int64 v10; // r12
  _QWORD *v11; // rax
  unsigned __int64 v12; // r13
  __int64 v13; // rcx
  __int64 v14; // rdx
  _QWORD *v15; // rax
  _QWORD *v16; // rdx
  _BOOL8 v17; // rdi
  unsigned __int64 v18; // rax
  __int64 v19; // rdx
  __int64 *v20; // rbx
  __int64 *v21; // r13
  __int64 v22; // r12
  _QWORD *v23; // rax
  int i; // edx
  int v25; // ecx
  unsigned int v26; // esi
  unsigned __int64 **v27; // rdx
  unsigned __int64 *v28; // r10
  unsigned __int64 *v29; // rax
  unsigned __int64 *v30; // rdi
  __int64 v31; // rax
  _QWORD *v32; // rax
  _QWORD *v33; // r8
  __int64 v34; // r13
  unsigned __int64 v35; // rdi
  _QWORD *v36; // rdx
  __int64 v37; // rsi
  __int64 v38; // rcx
  unsigned __int64 **v39; // r14
  _QWORD *v40; // rsi
  unsigned __int64 v41; // rdi
  __int64 v42; // rcx
  __int64 v43; // rdx
  __int64 v44; // r12
  _QWORD *v45; // rbx
  _QWORD *v46; // r14
  __int64 v47; // rax
  __int64 v48; // rsi
  __int64 v49; // rdi
  unsigned int v50; // ecx
  __int64 *v51; // rdx
  __int64 v52; // r9
  __int64 v53; // rax
  unsigned int v54; // edx
  __int64 v55; // rdx
  _QWORD *v56; // rsi
  unsigned __int64 v57; // rdi
  __int64 v58; // rcx
  __int64 v59; // rdx
  _QWORD *v60; // r12
  int v62; // edx
  int v63; // r10d
  __int64 v64; // rdx
  __int64 v65; // rcx
  __int64 v66; // r9
  _QWORD *v67; // r10
  int v68; // r8d
  unsigned __int64 *v69; // rdi
  __int64 v70; // r15
  _QWORD *v71; // r14
  unsigned int v72; // edx
  unsigned __int64 v73; // rbx
  __int64 v74; // rdx
  unsigned __int64 v75; // r11
  unsigned __int64 *v76; // rdx
  int v77; // r12d
  unsigned __int64 *v78; // rsi
  unsigned __int64 *v79; // rcx
  unsigned __int64 v80; // rdx
  _QWORD *v81; // rax
  _QWORD *v82; // rsi
  __int64 v83; // rdi
  __int64 v84; // rcx
  _QWORD *v86; // [rsp+8h] [rbp-4D8h]
  __int64 v87; // [rsp+10h] [rbp-4D0h]
  __int64 v88; // [rsp+18h] [rbp-4C8h]
  int v89; // [rsp+24h] [rbp-4BCh]
  __int64 v90; // [rsp+28h] [rbp-4B8h]
  unsigned __int64 v91; // [rsp+28h] [rbp-4B8h]
  __int64 v92; // [rsp+40h] [rbp-4A0h]
  _QWORD *v93; // [rsp+50h] [rbp-490h]
  _QWORD *v94; // [rsp+58h] [rbp-488h]
  _QWORD *v95; // [rsp+60h] [rbp-480h]
  _QWORD *v96; // [rsp+60h] [rbp-480h]
  unsigned __int64 **v97; // [rsp+60h] [rbp-480h]
  unsigned int v98; // [rsp+68h] [rbp-478h]
  unsigned __int64 **v99; // [rsp+68h] [rbp-478h]
  _QWORD *v101; // [rsp+78h] [rbp-468h]
  unsigned __int64 v102; // [rsp+88h] [rbp-458h] BYREF
  _QWORD *v103; // [rsp+90h] [rbp-450h] BYREF
  __int64 v104; // [rsp+98h] [rbp-448h]
  _QWORD v105[64]; // [rsp+A0h] [rbp-440h] BYREF
  unsigned __int64 *v106; // [rsp+2A0h] [rbp-240h] BYREF
  unsigned int v107; // [rsp+2A8h] [rbp-238h]
  unsigned int v108; // [rsp+2ACh] [rbp-234h]
  _QWORD v109[70]; // [rsp+2B0h] [rbp-230h] BYREF

  v101 = a1 + 2;
  if ( a1[7] == a1[8] )
  {
    v80 = *a3;
    v81 = (_QWORD *)a1[3];
    v82 = a1 + 2;
    v103 = (_QWORD *)v80;
    if ( !v81 )
      goto LABEL_114;
    do
    {
      while ( 1 )
      {
        v83 = v81[2];
        v84 = v81[3];
        if ( v81[4] >= v80 )
          break;
        v81 = (_QWORD *)v81[3];
        if ( !v84 )
          goto LABEL_112;
      }
      v82 = v81;
      v81 = (_QWORD *)v81[2];
    }
    while ( v83 );
LABEL_112:
    if ( v82 == v101 || v82[4] > v80 )
    {
LABEL_114:
      v106 = (unsigned __int64 *)&v103;
      v82 = (_QWORD *)sub_1C12C30(a1 + 1, v82, &v106);
    }
    return v82 + 5;
  }
  else
  {
    v7 = 0;
    v8 = v105;
    v98 = 0;
    v105[0] = a3;
    v104 = 0x4000000001LL;
    v103 = v105;
    v93 = a1 + 1;
    while ( 1 )
    {
      v9 = (unsigned __int64 *)v8[v7];
      v10 = (__int64)v101;
      v11 = (_QWORD *)a1[3];
      v12 = *v9;
      if ( !v11 )
        goto LABEL_10;
      do
      {
        while ( 1 )
        {
          v13 = v11[2];
          v14 = v11[3];
          if ( v11[4] >= v12 )
            break;
          v11 = (_QWORD *)v11[3];
          if ( !v14 )
            goto LABEL_8;
        }
        v10 = (__int64)v11;
        v11 = (_QWORD *)v11[2];
      }
      while ( v13 );
LABEL_8:
      if ( (_QWORD *)v10 == v101 || *(_QWORD *)(v10 + 32) > v12 )
      {
LABEL_10:
        v95 = (_QWORD *)v10;
        v10 = sub_22077B0(88);
        *(_QWORD *)(v10 + 32) = v12;
        *(_DWORD *)(v10 + 48) = 0;
        *(_QWORD *)(v10 + 56) = 0;
        *(_QWORD *)(v10 + 64) = v10 + 48;
        *(_QWORD *)(v10 + 72) = v10 + 48;
        *(_QWORD *)(v10 + 80) = 0;
        v15 = sub_13BFCF0(v93, v95, (unsigned __int64 *)(v10 + 32));
        if ( v16 )
        {
          v17 = v15 || v101 == v16 || v12 < v16[4];
          sub_220F040(v17, v10, v16, v101);
          ++a1[6];
        }
        else
        {
          v96 = v15;
          sub_1C126B0(0);
          j_j___libc_free_0(v10, 88);
          v10 = (__int64)v96;
        }
      }
      if ( v12 )
      {
        while ( 1 )
        {
          v12 = *(_QWORD *)(v12 + 8);
          if ( !v12 )
            break;
          v23 = sub_1648700(v12);
          if ( (unsigned __int8)(*((_BYTE *)v23 + 16) - 25) <= 9u )
          {
LABEL_34:
            v30 = (unsigned __int64 *)v23[5];
            v31 = *(unsigned int *)(a2 + 72);
            v106 = v30;
            if ( (_DWORD)v31 )
            {
              a6 = v31 - 1;
              a5 = *(_QWORD *)(a2 + 56);
              v26 = (v31 - 1) & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
              v27 = (unsigned __int64 **)(a5 + 16LL * v26);
              v28 = *v27;
              if ( v30 != *v27 )
              {
                for ( i = 1; ; i = v25 )
                {
                  if ( v28 == (unsigned __int64 *)-8LL )
                    goto LABEL_32;
                  v25 = i + 1;
                  v26 = a6 & (i + v26);
                  v27 = (unsigned __int64 **)(a5 + 16LL * v26);
                  v28 = *v27;
                  if ( v30 == *v27 )
                    break;
                }
              }
              if ( v27 != (unsigned __int64 **)(a5 + 16 * v31) )
              {
                v29 = v27[1];
                if ( v29 )
                {
                  if ( v9 != (unsigned __int64 *)v29[1] )
                    sub_1444990((_QWORD *)(v10 + 40), (unsigned __int64 *)&v106);
                }
              }
            }
LABEL_32:
            while ( 1 )
            {
              v12 = *(_QWORD *)(v12 + 8);
              if ( !v12 )
                goto LABEL_16;
              v23 = sub_1648700(v12);
              if ( (unsigned __int8)(*((_BYTE *)v23 + 16) - 25) <= 9u )
                goto LABEL_34;
            }
          }
        }
      }
LABEL_16:
      v18 = v9[4];
      v19 = (unsigned int)v104;
      if ( v18 != v9[3] )
      {
        v20 = (__int64 *)v9[3];
        v21 = (__int64 *)v18;
        do
        {
          v22 = *v20;
          if ( HIDWORD(v104) <= (unsigned int)v19 )
          {
            sub_16CD150((__int64)&v103, v105, 0, 8, a5, a6);
            v19 = (unsigned int)v104;
          }
          ++v20;
          v103[v19] = v22;
          v19 = (unsigned int)(v104 + 1);
          LODWORD(v104) = v104 + 1;
        }
        while ( v21 != v20 );
      }
      v7 = v98 + 1;
      if ( (_DWORD)v7 == (_DWORD)v19 )
        break;
      ++v98;
      v8 = v103;
    }
    v32 = (_QWORD *)a1[3];
    if ( (_DWORD)v19 )
    {
      v92 = v98;
LABEL_40:
      v33 = v101;
      v34 = v103[v92];
      v35 = *(_QWORD *)v34;
      v102 = *(_QWORD *)v34;
      if ( !v32 )
        goto LABEL_47;
      v36 = v32;
      do
      {
        while ( 1 )
        {
          v37 = v36[2];
          v38 = v36[3];
          if ( v36[4] >= v35 )
            break;
          v36 = (_QWORD *)v36[3];
          if ( !v38 )
            goto LABEL_45;
        }
        v33 = v36;
        v36 = (_QWORD *)v36[2];
      }
      while ( v37 );
LABEL_45:
      if ( v33 == v101 || v33[4] > v35 )
      {
LABEL_47:
        v106 = &v102;
        v33 = (_QWORD *)sub_1C12C30(a1 + 1, v33, &v106);
        v32 = (_QWORD *)a1[3];
      }
      v39 = *(unsigned __int64 ***)(v34 + 24);
      v94 = v33 + 5;
      v97 = *(unsigned __int64 ***)(v34 + 32);
      if ( v97 == v39 )
        goto LABEL_76;
LABEL_49:
      v40 = v101;
      v41 = **v39;
      v102 = v41;
      if ( !v32 )
        goto LABEL_56;
      do
      {
        while ( 1 )
        {
          v42 = v32[2];
          v43 = v32[3];
          if ( v32[4] >= v41 )
            break;
          v32 = (_QWORD *)v32[3];
          if ( !v43 )
            goto LABEL_54;
        }
        v40 = v32;
        v32 = (_QWORD *)v32[2];
      }
      while ( v42 );
LABEL_54:
      if ( v40 == v101 || v40[4] > v41 )
      {
LABEL_56:
        v106 = &v102;
        v40 = (_QWORD *)sub_1C12C30(a1 + 1, v40, &v106);
      }
      v44 = v40[8];
      v45 = v40 + 6;
      if ( v40 + 6 == (_QWORD *)v44 )
        goto LABEL_75;
      v99 = v39;
      v46 = v94;
      while ( 1 )
      {
        v47 = *(unsigned int *)(a2 + 72);
        if ( !(_DWORD)v47 )
          goto LABEL_95;
        v48 = *(_QWORD *)(v44 + 32);
        v49 = *(_QWORD *)(a2 + 56);
        v50 = (v47 - 1) & (((unsigned int)v48 >> 9) ^ ((unsigned int)v48 >> 4));
        v51 = (__int64 *)(v49 + 16LL * v50);
        v52 = *v51;
        if ( v48 != *v51 )
          break;
LABEL_61:
        if ( v51 == (__int64 *)(v49 + 16 * v47) )
          goto LABEL_95;
        v53 = v51[1];
        if ( !v53 || v34 == v53 )
          goto LABEL_95;
        if ( v34 != *(_QWORD *)(v53 + 8) )
        {
          if ( v53 != *(_QWORD *)(v34 + 8) && *(_DWORD *)(v34 + 16) < *(_DWORD *)(v53 + 16) )
          {
            if ( !*(_BYTE *)(a2 + 96) )
            {
              v54 = *(_DWORD *)(a2 + 100) + 1;
              *(_DWORD *)(a2 + 100) = v54;
              if ( v54 <= 0x20 )
              {
                do
                {
                  v55 = v53;
                  v53 = *(_QWORD *)(v53 + 8);
                }
                while ( v53 && *(_DWORD *)(v34 + 16) <= *(_DWORD *)(v53 + 16) );
                if ( v34 == v55 )
                  goto LABEL_73;
                goto LABEL_95;
              }
              v64 = *(_QWORD *)(a2 + 80);
              v108 = 32;
              v106 = v109;
              if ( v64 )
              {
                v65 = *(_QWORD *)(v64 + 24);
                v66 = a2;
                v67 = v46;
                v68 = 1;
                v109[0] = v64;
                v69 = v109;
                v70 = v44;
                v71 = v45;
                v109[1] = v65;
                v107 = 1;
                *(_DWORD *)(v64 + 48) = 0;
                v72 = 1;
                do
                {
                  v77 = v68++;
                  v78 = &v69[2 * v72 - 2];
                  v79 = (unsigned __int64 *)v78[1];
                  if ( v79 == *(unsigned __int64 **)(*v78 + 32) )
                  {
                    --v72;
                    *(_DWORD *)(*v78 + 52) = v77;
                    v107 = v72;
                  }
                  else
                  {
                    v73 = *v79;
                    v78[1] = (unsigned __int64)(v79 + 1);
                    v74 = v107;
                    v75 = *(_QWORD *)(v73 + 24);
                    if ( v107 >= v108 )
                    {
                      v86 = v67;
                      v87 = v66;
                      v88 = v53;
                      v89 = v68;
                      v91 = *(_QWORD *)(v73 + 24);
                      sub_16CD150((__int64)&v106, v109, 0, 16, v68, v66);
                      v69 = v106;
                      v74 = v107;
                      v67 = v86;
                      v66 = v87;
                      v53 = v88;
                      v68 = v89;
                      v75 = v91;
                    }
                    v76 = &v69[2 * v74];
                    *v76 = v73;
                    v76[1] = v75;
                    v72 = ++v107;
                    *(_DWORD *)(v73 + 48) = v77;
                    v69 = v106;
                  }
                }
                while ( v72 );
                v45 = v71;
                v44 = v70;
                *(_BYTE *)(v66 + 96) = 1;
                a2 = v66;
                *(_DWORD *)(v66 + 100) = 0;
                v46 = v67;
                if ( v69 != v109 )
                {
                  v90 = v53;
                  _libc_free((unsigned __int64)v69);
                  v53 = v90;
                }
              }
            }
            if ( *(_DWORD *)(v53 + 48) >= *(_DWORD *)(v34 + 48) && *(_DWORD *)(v53 + 52) <= *(_DWORD *)(v34 + 52) )
              goto LABEL_73;
          }
LABEL_95:
          sub_1444990(v46, (unsigned __int64 *)(v44 + 32));
        }
LABEL_73:
        v44 = sub_220EF30(v44);
        if ( v45 == (_QWORD *)v44 )
        {
          v39 = v99;
LABEL_75:
          ++v39;
          v32 = (_QWORD *)a1[3];
          if ( v97 == v39 )
          {
LABEL_76:
            if ( --v92 == -1 )
              goto LABEL_77;
            goto LABEL_40;
          }
          goto LABEL_49;
        }
      }
      v62 = 1;
      while ( v52 != -8 )
      {
        v63 = v62 + 1;
        v50 = (v47 - 1) & (v62 + v50);
        v51 = (__int64 *)(v49 + 16LL * v50);
        v52 = *v51;
        if ( v48 == *v51 )
          goto LABEL_61;
        v62 = v63;
      }
      goto LABEL_95;
    }
LABEL_77:
    v56 = v101;
    v57 = *a3;
    v102 = *a3;
    if ( !v32 )
      goto LABEL_84;
    do
    {
      while ( 1 )
      {
        v58 = v32[2];
        v59 = v32[3];
        if ( v32[4] >= v57 )
          break;
        v32 = (_QWORD *)v32[3];
        if ( !v59 )
          goto LABEL_82;
      }
      v56 = v32;
      v32 = (_QWORD *)v32[2];
    }
    while ( v58 );
LABEL_82:
    if ( v101 == v56 || v56[4] > v57 )
    {
LABEL_84:
      v106 = &v102;
      v56 = (_QWORD *)sub_1C12C30(a1 + 1, v56, &v106);
    }
    v60 = v56 + 5;
    if ( v103 != v105 )
      _libc_free((unsigned __int64)v103);
  }
  return v60;
}
