// Function: sub_2EC6D00
// Address: 0x2ec6d00
//
__int64 __fastcall sub_2EC6D00(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v4; // rbx
  __int64 v5; // r12
  __int64 v6; // r14
  int v7; // r13d
  __int64 v8; // r9
  unsigned __int8 v9; // r11
  unsigned int v10; // r8d
  unsigned int v11; // edx
  __int64 v12; // rsi
  _DWORD *v13; // rdi
  __int64 v14; // rcx
  unsigned __int8 v15; // r15
  __int64 v16; // r12
  __int64 v17; // rbx
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // r11
  __int64 v21; // r15
  unsigned int v22; // eax
  __int64 v23; // rsi
  __int64 *v24; // r8
  int v25; // r9d
  __int64 v26; // r10
  __int64 v27; // rax
  __int64 v28; // rcx
  unsigned __int64 v29; // rax
  __int64 k; // rdi
  __int16 v31; // dx
  __int64 v32; // rsi
  unsigned int v33; // ecx
  unsigned int v34; // r11d
  __int64 *v35; // rdx
  __int64 v36; // rdi
  unsigned __int64 v37; // r15
  __int64 *v38; // rax
  __int64 *v39; // r8
  __int64 v40; // r10
  int v41; // r9d
  __int64 v42; // rcx
  unsigned int v43; // edx
  unsigned int v44; // r10d
  __int64 v45; // rsi
  _DWORD *v46; // rdi
  __int64 v47; // rcx
  __int64 v48; // r12
  __int64 *v49; // rbx
  __int64 v50; // r13
  __int64 v51; // rax
  __int64 v52; // r15
  unsigned __int64 v53; // rsi
  __int64 v54; // rcx
  unsigned __int64 i; // rax
  __int64 j; // rdi
  __int16 v57; // dx
  __int64 v58; // rsi
  __int64 v59; // rdi
  unsigned int v60; // ecx
  __int64 *v61; // rdx
  __int64 v62; // r9
  __int64 v63; // rdi
  __int64 *v64; // rcx
  int v65; // r8d
  int v66; // r9d
  __int64 v67; // rsi
  int v68; // edx
  unsigned int v69; // esi
  __int64 v70; // rsi
  __int64 v71; // rax
  __int64 v72; // rax
  __int64 v73; // rdx
  int v74; // eax
  signed __int64 v75; // r15
  __int64 *v76; // rax
  int v77; // edx
  __int64 v78; // rcx
  __int64 v79; // rax
  __int64 v80; // r8
  _QWORD *v81; // rdx
  _QWORD *v82; // rdi
  int v83; // r8d
  int v84; // [rsp+8h] [rbp-78h]
  int v85; // [rsp+8h] [rbp-78h]
  int v86; // [rsp+8h] [rbp-78h]
  __int64 v87; // [rsp+8h] [rbp-78h]
  __int64 v88; // [rsp+10h] [rbp-70h]
  __int64 v89; // [rsp+10h] [rbp-70h]
  __int64 v90; // [rsp+10h] [rbp-70h]
  __int64 v91; // [rsp+10h] [rbp-70h]
  __int64 v92; // [rsp+18h] [rbp-68h]
  _QWORD *v93; // [rsp+18h] [rbp-68h]
  __int64 v94; // [rsp+20h] [rbp-60h]
  __int64 *v95; // [rsp+20h] [rbp-60h]
  unsigned __int64 v96; // [rsp+20h] [rbp-60h]
  __int64 *v97; // [rsp+20h] [rbp-60h]
  int v98; // [rsp+20h] [rbp-60h]
  __int64 v99; // [rsp+20h] [rbp-60h]
  __int64 v100; // [rsp+28h] [rbp-58h]

  result = a2 + 24 * a3;
  v100 = result;
  if ( result != a2 )
  {
    v4 = a1;
    v5 = a2;
    v6 = a1 + 328;
    while ( 1 )
    {
      v7 = *(_DWORD *)v5;
      if ( *(int *)v5 >= 0 )
        goto LABEL_20;
      v8 = v7 & 0x7FFFFFFF;
      v9 = *(_BYTE *)(v4 + 4017);
      if ( !v9 )
        break;
      if ( !*(_QWORD *)(v5 + 8) )
        v9 = *(_QWORD *)(v5 + 16) != 0;
      result = *(_QWORD *)(v4 + 3976);
      v10 = *(_DWORD *)(v4 + 3648);
      v11 = *(unsigned __int8 *)(result + v8);
      if ( v11 < v10 )
      {
        v12 = *(_QWORD *)(v4 + 3640);
        while ( 1 )
        {
          result = v11;
          v13 = (_DWORD *)(v12 + 40LL * v11);
          if ( (*v13 & 0x7FFFFFFF) == (_DWORD)v8 )
          {
            v14 = (unsigned int)v13[8];
            if ( (_DWORD)v14 != -1 && *(_DWORD *)(v12 + 40 * v14 + 36) == -1 )
              break;
          }
          v11 += 256;
          if ( v10 <= v11 )
            goto LABEL_20;
        }
        if ( v11 != -1 )
        {
          v94 = v5;
          v15 = v9;
          v16 = v4;
          do
          {
            v17 = 40 * result;
            v18 = v12 + 40 * result;
            v19 = *(_QWORD *)(v18 + 24);
            if ( (*(_BYTE *)(v19 + 249) & 4) == 0 && v19 != v6 )
            {
              sub_2F76A30(
                *(_DWORD *)(v16 + 4000) + (*(_DWORD *)(v19 + 200) << 6),
                v15,
                *(_QWORD *)(v16 + 40),
                0,
                v10,
                v8,
                v7,
                0,
                0);
              v12 = *(_QWORD *)(v16 + 3640);
              v18 = v12 + v17;
            }
            result = *(unsigned int *)(v18 + 36);
          }
          while ( (_DWORD)result != -1 );
          v4 = v16;
          v5 = v94;
        }
      }
LABEL_20:
      v5 += 24;
      if ( v100 == v5 )
        return result;
    }
    v20 = *(_QWORD *)(v4 + 3464);
    v21 = 8 * v8;
    v22 = *(_DWORD *)(v20 + 160);
    if ( v22 > (unsigned int)v8 && *(_QWORD *)(*(_QWORD *)(v20 + 152) + 8 * v8) )
    {
LABEL_24:
      v23 = sub_2EC2050(*(_QWORD *)(v4 + 6312), *(_QWORD *)(v4 + 904) + 48LL);
      v27 = *(_QWORD *)(v4 + 904);
      if ( v23 == v27 + 48 )
      {
        v73 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v4 + 3464) + 32LL) + 152LL)
                        + 16LL * *(unsigned int *)(v27 + 24)
                        + 8);
        v74 = (v73 >> 1) & 3;
        if ( v74 )
          v75 = v73 & 0xFFFFFFFFFFFFFFF8LL | (2LL * (v74 - 1));
        else
          v75 = *(_QWORD *)(v73 & 0xFFFFFFFFFFFFFFF8LL) & 0xFFFFFFFFFFFFFFF8LL | 6;
        v86 = v25;
        v90 = v26;
        v97 = v24;
        v76 = (__int64 *)sub_2E09D00(v24, v75);
        v39 = v97;
        v40 = v90;
        v92 = 0;
        v41 = v86;
        if ( v76 != (__int64 *)(*v97 + 24LL * *((unsigned int *)v97 + 2))
          && (*(_DWORD *)((*v76 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)((*v76 >> 1) & 3)) <= (*(_DWORD *)((v75 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v75 >> 1) & 3) )
        {
          v92 = v76[2];
        }
LABEL_37:
        result = *(_QWORD *)(v4 + 3976);
        v43 = *(unsigned __int8 *)(result + v40);
        v44 = *(_DWORD *)(v4 + 3648);
        if ( v43 >= v44 )
          goto LABEL_20;
        v45 = *(_QWORD *)(v4 + 3640);
        while ( 1 )
        {
          result = v43;
          v46 = (_DWORD *)(v45 + 40LL * v43);
          if ( (*v46 & 0x7FFFFFFF) == v41 )
          {
            v47 = (unsigned int)v46[8];
            if ( (_DWORD)v47 != -1 && *(_DWORD *)(v45 + 40 * v47 + 36) == -1 )
              break;
          }
          v43 += 256;
          if ( v44 <= v43 )
            goto LABEL_20;
        }
        if ( v43 == -1 )
          goto LABEL_20;
        v85 = v7;
        v89 = v5;
        v48 = v4;
        v49 = v39;
        while ( 1 )
        {
          v50 = 40 * result;
          v51 = v45 + 40 * result;
          v52 = *(_QWORD *)(v51 + 24);
          if ( (*(_BYTE *)(v52 + 249) & 4) == 0 && v52 != v6 )
            break;
LABEL_65:
          result = *(unsigned int *)(v51 + 36);
          if ( (_DWORD)result == -1 )
          {
            v4 = v48;
            v5 = v89;
            goto LABEL_20;
          }
        }
        v53 = *(_QWORD *)v52;
        v54 = *(_QWORD *)(*(_QWORD *)(v48 + 3464) + 32LL);
        for ( i = *(_QWORD *)v52; (*(_BYTE *)(i + 44) & 4) != 0; i = *(_QWORD *)i & 0xFFFFFFFFFFFFFFF8LL )
          ;
        if ( (*(_DWORD *)(*(_QWORD *)v52 + 44LL) & 8) != 0 )
        {
          do
            v53 = *(_QWORD *)(v53 + 8);
          while ( (*(_BYTE *)(v53 + 44) & 8) != 0 );
        }
        for ( j = *(_QWORD *)(v53 + 8); j != i; i = *(_QWORD *)(i + 8) )
        {
          v57 = *(_WORD *)(i + 68);
          if ( (unsigned __int16)(v57 - 14) > 4u && v57 != 24 )
            break;
        }
        v58 = *(unsigned int *)(v54 + 144);
        v59 = *(_QWORD *)(v54 + 128);
        if ( (_DWORD)v58 )
        {
          v60 = (v58 - 1) & (((unsigned int)i >> 9) ^ ((unsigned int)i >> 4));
          v61 = (__int64 *)(v59 + 16LL * v60);
          v62 = *v61;
          if ( *v61 == i )
          {
LABEL_57:
            v96 = v61[1] & 0xFFFFFFFFFFFFFFF8LL;
            v63 = 0;
            v64 = (__int64 *)sub_2E09D00(v49, v96);
            v67 = *v49 + 24LL * *((unsigned int *)v49 + 2);
            if ( v64 != (__int64 *)v67 )
            {
              v63 = 0;
              if ( (*(_DWORD *)((*v64 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v64 >> 1) & 3) <= *(_DWORD *)(v96 + 24) )
              {
                v63 = v64[2];
                if ( (v96 != (v64[1] & 0xFFFFFFFFFFFFFFF8LL) || (__int64 *)v67 != v64 + 3)
                  && v96 == *(_QWORD *)(v63 + 8) )
                {
                  v63 = 0;
                }
              }
            }
            if ( v92 == v63 )
              sub_2F76A30(
                *(_DWORD *)(v48 + 4000) + (*(_DWORD *)(v52 + 200) << 6),
                1,
                *(_QWORD *)(v48 + 40),
                0,
                v65,
                v66,
                v85,
                0,
                0);
            v45 = *(_QWORD *)(v48 + 3640);
            v51 = v45 + v50;
            goto LABEL_65;
          }
          v68 = 1;
          while ( v62 != -4096 )
          {
            v83 = v68 + 1;
            v60 = (v58 - 1) & (v68 + v60);
            v61 = (__int64 *)(v59 + 16LL * v60);
            v62 = *v61;
            if ( i == *v61 )
              goto LABEL_57;
            v68 = v83;
          }
        }
        v61 = (__int64 *)(v59 + 16 * v58);
        goto LABEL_57;
      }
      v28 = *(_QWORD *)(*(_QWORD *)(v4 + 3464) + 32LL);
      v29 = v23;
      if ( (*(_DWORD *)(v23 + 44) & 4) != 0 )
      {
        do
          v29 = *(_QWORD *)v29 & 0xFFFFFFFFFFFFFFF8LL;
        while ( (*(_BYTE *)(v29 + 44) & 4) != 0 );
      }
      for ( ; (*(_BYTE *)(v23 + 44) & 8) != 0; v23 = *(_QWORD *)(v23 + 8) )
        ;
      for ( k = *(_QWORD *)(v23 + 8); k != v29; v29 = *(_QWORD *)(v29 + 8) )
      {
        v31 = *(_WORD *)(v29 + 68);
        if ( (unsigned __int16)(v31 - 14) > 4u && v31 != 24 )
          break;
      }
      v32 = *(_QWORD *)(v28 + 128);
      v33 = *(_DWORD *)(v28 + 144);
      if ( v33 )
      {
        v34 = (v33 - 1) & (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4));
        v35 = (__int64 *)(v32 + 16LL * v34);
        v36 = *v35;
        if ( v29 == *v35 )
        {
LABEL_35:
          v84 = v25;
          v88 = v26;
          v37 = v35[1] & 0xFFFFFFFFFFFFFFF8LL;
          v95 = v24;
          v38 = (__int64 *)sub_2E09D00(v24, v37);
          v39 = v95;
          v40 = v88;
          v92 = 0;
          v41 = v84;
          v42 = *v95 + 24LL * *((unsigned int *)v95 + 2);
          if ( v38 != (__int64 *)v42
            && (*(_DWORD *)((*v38 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v38 >> 1) & 3) <= *(_DWORD *)(v37 + 24) )
          {
            v92 = v38[2];
            if ( v37 != (v38[1] & 0xFFFFFFFFFFFFFFF8LL) || (__int64 *)v42 != v38 + 3 )
            {
              v72 = 0;
              if ( v37 != *(_QWORD *)(v92 + 8) )
                v72 = v92;
              v92 = v72;
            }
          }
          goto LABEL_37;
        }
        v77 = 1;
        while ( v36 != -4096 )
        {
          v34 = (v33 - 1) & (v77 + v34);
          v98 = v77 + 1;
          v35 = (__int64 *)(v32 + 16LL * v34);
          v36 = *v35;
          if ( *v35 == v29 )
            goto LABEL_35;
          v77 = v98;
        }
      }
      v35 = (__int64 *)(v32 + 16LL * v33);
      goto LABEL_35;
    }
    v69 = v8 + 1;
    if ( v22 < (int)v8 + 1 )
    {
      v78 = v22;
      if ( v69 != (unsigned __int64)v22 )
      {
        if ( v69 >= (unsigned __int64)v22 )
        {
          v79 = *(_QWORD *)(v20 + 168);
          v80 = v69 - v78;
          if ( v69 > (unsigned __int64)*(unsigned int *)(v20 + 164) )
          {
            v87 = v69 - v78;
            v91 = *(_QWORD *)(v20 + 168);
            v99 = *(_QWORD *)(v4 + 3464);
            sub_C8D5F0(v20 + 152, (const void *)(v20 + 168), v69, 8u, v80, v8);
            v20 = v99;
            v80 = v87;
            v79 = v91;
            v78 = *(unsigned int *)(v99 + 160);
          }
          v70 = *(_QWORD *)(v20 + 152);
          v81 = (_QWORD *)(v70 + 8 * v78);
          v82 = &v81[v80];
          if ( v81 != v82 )
          {
            do
              *v81++ = v79;
            while ( v82 != v81 );
            LODWORD(v78) = *(_DWORD *)(v20 + 160);
            v70 = *(_QWORD *)(v20 + 152);
          }
          *(_DWORD *)(v20 + 160) = v80 + v78;
          goto LABEL_72;
        }
        *(_DWORD *)(v20 + 160) = v69;
      }
    }
    v70 = *(_QWORD *)(v20 + 152);
LABEL_72:
    v93 = (_QWORD *)v20;
    v71 = sub_2E10F30(v7);
    *(_QWORD *)(v70 + v21) = v71;
    sub_2E11E80(v93, v71);
    goto LABEL_24;
  }
  return result;
}
