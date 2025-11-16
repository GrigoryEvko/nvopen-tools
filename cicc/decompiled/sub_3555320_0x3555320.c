// Function: sub_3555320
// Address: 0x3555320
//
__int64 __fastcall sub_3555320(__int64 a1, __int64 *a2, __int64 *a3, __int64 a4)
{
  __int64 *v4; // rbx
  __int64 *v5; // r12
  __int64 *v7; // rsi
  unsigned __int64 v8; // rcx
  __int64 *v9; // r12
  _QWORD *v10; // rax
  unsigned int v11; // esi
  __int64 v12; // r8
  int v13; // r11d
  __int64 v14; // r9
  unsigned int v15; // ecx
  __int64 v16; // rax
  __int64 v17; // r10
  __int64 v18; // rdx
  unsigned int v19; // ecx
  __int64 v20; // rsi
  int v21; // eax
  unsigned int v22; // r11d
  __int64 v23; // r15
  __int64 v24; // r14
  __int64 v25; // rax
  unsigned int v26; // r11d
  __int64 v27; // rbx
  __int64 v28; // r13
  unsigned int v29; // r10d
  unsigned int v30; // esi
  unsigned int v31; // edi
  __int64 v32; // rcx
  __int64 *v33; // rax
  int v34; // r9d
  unsigned int v35; // r8d
  _QWORD *v36; // r12
  __int64 v37; // rdx
  _DWORD *v38; // r12
  unsigned int v39; // r9d
  __int64 *v40; // rax
  __int64 v41; // r8
  unsigned int v42; // ecx
  unsigned int *v43; // rdx
  unsigned int v44; // eax
  __int64 v45; // rax
  __int64 v46; // rbx
  __int64 v47; // r13
  __int64 v48; // rax
  __int64 *v49; // rax
  unsigned int v50; // edx
  __int64 *v51; // rax
  __int64 v52; // rdi
  __int64 v53; // rsi
  int v55; // edx
  unsigned int v56; // esi
  int v57; // eax
  __int64 *v58; // rdx
  __int64 v59; // rdi
  unsigned int v60; // esi
  __int64 v61; // rdi
  __int64 *v62; // r9
  unsigned int v63; // esi
  __int64 v64; // rdi
  int v65; // r12d
  __int64 *v66; // r9
  int v67; // r12d
  unsigned int v68; // esi
  __int64 v69; // rdi
  _QWORD *v70; // rax
  int v71; // r10d
  __int64 v72; // rdi
  unsigned int v73; // [rsp+8h] [rbp-98h]
  unsigned int v74; // [rsp+8h] [rbp-98h]
  unsigned int v75; // [rsp+8h] [rbp-98h]
  unsigned int v76; // [rsp+8h] [rbp-98h]
  _QWORD *v78; // [rsp+20h] [rbp-80h]
  unsigned int v79; // [rsp+28h] [rbp-78h]
  unsigned int v81; // [rsp+38h] [rbp-68h]
  int v82; // [rsp+38h] [rbp-68h]
  unsigned int v83; // [rsp+38h] [rbp-68h]
  int v84; // [rsp+38h] [rbp-68h]
  unsigned int v85; // [rsp+38h] [rbp-68h]
  unsigned int v86; // [rsp+38h] [rbp-68h]
  int v87; // [rsp+38h] [rbp-68h]
  unsigned int v88; // [rsp+3Ch] [rbp-64h]
  unsigned int v89; // [rsp+3Ch] [rbp-64h]
  unsigned int v90; // [rsp+3Ch] [rbp-64h]
  __int64 v91; // [rsp+40h] [rbp-60h] BYREF
  __int64 v92; // [rsp+48h] [rbp-58h] BYREF
  __int64 v93; // [rsp+50h] [rbp-50h] BYREF
  __int64 v94; // [rsp+58h] [rbp-48h]
  __int64 v95; // [rsp+60h] [rbp-40h]
  unsigned int v96; // [rsp+68h] [rbp-38h]

  v4 = (__int64 *)(a1 + 48);
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = a1 + 48;
  *(_QWORD *)(a1 + 40) = 0;
  if ( a2 == a3 )
  {
    v93 = 0;
    v8 = 0;
    v94 = 0;
    *(_BYTE *)(a1 + 48) = 1;
    *(_QWORD *)(a1 + 52) = 0;
    *(_QWORD *)(a1 + 60) = 0;
    *(_QWORD *)(a1 + 72) = 0;
    *(_DWORD *)(a1 + 80) = 0;
    v95 = 0;
    v70 = *(_QWORD **)(a4 + 3464);
    v96 = 0;
    v78 = v70;
  }
  else
  {
    v5 = a2;
    do
    {
      v7 = v5++;
      sub_3554C70(a1, v7);
    }
    while ( a3 != v5 );
    v93 = 0;
    v94 = 0;
    v8 = *(unsigned int *)(a1 + 40);
    v9 = *(__int64 **)(a1 + 32);
    *(_BYTE *)(a1 + 48) = 1;
    *(_QWORD *)(a1 + 52) = 0;
    *(_QWORD *)(a1 + 60) = 0;
    v4 = &v9[v8];
    *(_QWORD *)(a1 + 72) = 0;
    *(_DWORD *)(a1 + 80) = 0;
    v79 = v8;
    v10 = *(_QWORD **)(a4 + 3464);
    v95 = 0;
    v96 = 0;
    v78 = v10;
    if ( v9 != v4 )
    {
      v11 = 0;
      v12 = 0;
      while ( 1 )
      {
        v18 = *v9;
        v91 = *v9;
        if ( v11 )
        {
          v13 = 1;
          v14 = 0;
          v15 = (v11 - 1) & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
          v16 = v12 + 16LL * v15;
          v17 = *(_QWORD *)v16;
          if ( v18 == *(_QWORD *)v16 )
          {
LABEL_7:
            ++v9;
            *(_DWORD *)(v16 + 8) = 0;
            if ( v4 == v9 )
              goto LABEL_17;
            goto LABEL_8;
          }
          while ( v17 != -4096 )
          {
            if ( !v14 && v17 == -8192 )
              v14 = v16;
            v15 = (v11 - 1) & (v13 + v15);
            v16 = v12 + 16LL * v15;
            v17 = *(_QWORD *)v16;
            if ( v18 == *(_QWORD *)v16 )
              goto LABEL_7;
            ++v13;
          }
          if ( !v14 )
            v14 = v16;
          ++v93;
          v21 = v95 + 1;
          v92 = v14;
          if ( 4 * ((int)v95 + 1) < 3 * v11 )
          {
            if ( v11 - (v21 + HIDWORD(v95)) <= v11 >> 3 )
            {
              sub_354B7C0((__int64)&v93, v11);
              sub_3546CD0((__int64)&v93, &v91, &v92);
              v18 = v91;
              v14 = v92;
              v21 = v95 + 1;
            }
            goto LABEL_14;
          }
        }
        else
        {
          ++v93;
          v92 = 0;
        }
        sub_354B7C0((__int64)&v93, 2 * v11);
        if ( v96 )
        {
          v18 = v91;
          v19 = (v96 - 1) & (((unsigned int)v91 >> 9) ^ ((unsigned int)v91 >> 4));
          v14 = v94 + 16LL * v19;
          v20 = *(_QWORD *)v14;
          if ( v91 == *(_QWORD *)v14 )
          {
LABEL_13:
            v92 = v14;
            v21 = v95 + 1;
          }
          else
          {
            v71 = 1;
            v72 = 0;
            while ( v20 != -4096 )
            {
              if ( !v72 && v20 == -8192 )
                v72 = v14;
              v19 = (v96 - 1) & (v71 + v19);
              v14 = v94 + 16LL * v19;
              v20 = *(_QWORD *)v14;
              if ( v91 == *(_QWORD *)v14 )
                goto LABEL_13;
              ++v71;
            }
            if ( !v72 )
              v72 = v14;
            v21 = v95 + 1;
            v92 = v72;
            v14 = v72;
          }
        }
        else
        {
          v92 = 0;
          v14 = 0;
          v18 = v91;
          v21 = v95 + 1;
        }
LABEL_14:
        LODWORD(v95) = v21;
        if ( *(_QWORD *)v14 != -4096 )
          --HIDWORD(v95);
        ++v9;
        *(_QWORD *)v14 = v18;
        *(_DWORD *)(v14 + 8) = 0;
        *(_DWORD *)(v14 + 8) = 0;
        if ( v4 == v9 )
        {
LABEL_17:
          v8 = *(unsigned int *)(a1 + 40);
          v4 = *(__int64 **)(a1 + 32);
          v79 = *(_DWORD *)(a1 + 40);
          break;
        }
LABEL_8:
        v12 = v94;
        v11 = v96;
      }
    }
    if ( v79 )
    {
      v8 = v79;
      v22 = 1;
      while ( 1 )
      {
        v88 = v22;
        v23 = v4[v22 - 1];
        v24 = v4[v22 % v8];
        v25 = sub_3545E90(v78, v23);
        v26 = v88;
        v27 = *(_QWORD *)v25;
        v28 = *(_QWORD *)v25 + 32LL * *(unsigned int *)(v25 + 8);
        if ( *(_QWORD *)v25 != v28 )
          break;
LABEL_31:
        v22 = v26 + 1;
        v4 = *(__int64 **)(a1 + 32);
        v8 = *(unsigned int *)(a1 + 40);
        if ( v22 > v79 )
          goto LABEL_32;
      }
      v89 = ((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4);
      v29 = ((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4);
      while ( 1 )
      {
        while ( v24 != *(_QWORD *)v27 )
        {
LABEL_22:
          v27 += 32;
          if ( v28 == v27 )
            goto LABEL_31;
        }
        v30 = v96;
        if ( v96 )
        {
          v31 = v96 - 1;
          v32 = v94;
          v33 = 0;
          v34 = 1;
          v35 = (v96 - 1) & v89;
          v36 = (_QWORD *)(v94 + 16LL * v35);
          v37 = *v36;
          if ( v23 == *v36 )
          {
LABEL_26:
            v38 = v36 + 1;
            goto LABEL_27;
          }
          while ( v37 != -4096 )
          {
            if ( v37 == -8192 && !v33 )
              v33 = v36;
            v35 = v31 & (v34 + v35);
            v36 = (_QWORD *)(v94 + 16LL * v35);
            v37 = *v36;
            if ( v23 == *v36 )
              goto LABEL_26;
            ++v34;
          }
          if ( !v33 )
            v33 = v36;
          ++v93;
          v55 = v95 + 1;
          if ( 4 * ((int)v95 + 1) < 3 * v96 )
          {
            if ( v96 - HIDWORD(v95) - v55 > v96 >> 3 )
              goto LABEL_63;
            v76 = v29;
            v86 = v26;
            sub_354B7C0((__int64)&v93, v96);
            if ( !v96 )
            {
LABEL_137:
              LODWORD(v95) = v95 + 1;
              BUG();
            }
            v66 = 0;
            v26 = v86;
            v67 = 1;
            v68 = (v96 - 1) & v89;
            v29 = v76;
            v55 = v95 + 1;
            v33 = (__int64 *)(v94 + 16LL * v68);
            v69 = *v33;
            if ( v23 == *v33 )
              goto LABEL_63;
            while ( v69 != -4096 )
            {
              if ( v69 == -8192 && !v66 )
                v66 = v33;
              v68 = (v96 - 1) & (v67 + v68);
              v33 = (__int64 *)(v94 + 16LL * v68);
              v69 = *v33;
              if ( v23 == *v33 )
                goto LABEL_63;
              ++v67;
            }
            goto LABEL_91;
          }
        }
        else
        {
          ++v93;
        }
        v75 = v29;
        v85 = v26;
        sub_354B7C0((__int64)&v93, 2 * v96);
        if ( !v96 )
          goto LABEL_137;
        v26 = v85;
        v63 = (v96 - 1) & v89;
        v29 = v75;
        v55 = v95 + 1;
        v33 = (__int64 *)(v94 + 16LL * v63);
        v64 = *v33;
        if ( v23 == *v33 )
          goto LABEL_63;
        v65 = 1;
        v66 = 0;
        while ( v64 != -4096 )
        {
          if ( !v66 && v64 == -8192 )
            v66 = v33;
          v63 = (v96 - 1) & (v65 + v63);
          v33 = (__int64 *)(v94 + 16LL * v63);
          v64 = *v33;
          if ( v23 == *v33 )
            goto LABEL_63;
          ++v65;
        }
LABEL_91:
        if ( v66 )
          v33 = v66;
LABEL_63:
        LODWORD(v95) = v55;
        if ( *v33 != -4096 )
          --HIDWORD(v95);
        *v33 = v23;
        v38 = v33 + 1;
        *((_DWORD *)v33 + 2) = 0;
        v30 = v96;
        if ( !v96 )
        {
          ++v93;
          goto LABEL_67;
        }
        v32 = v94;
        v31 = v96 - 1;
LABEL_27:
        v39 = v31 & v29;
        v40 = (__int64 *)(v32 + 16LL * (v31 & v29));
        v41 = *v40;
        if ( v24 == *v40 )
        {
LABEL_28:
          v42 = *((_DWORD *)v40 + 2);
          v43 = (unsigned int *)(v40 + 1);
          goto LABEL_29;
        }
        v82 = 1;
        v58 = 0;
        while ( v41 != -4096 )
        {
          if ( v41 == -8192 && !v58 )
            v58 = v40;
          v39 = v31 & (v82 + v39);
          v40 = (__int64 *)(v32 + 16LL * v39);
          v41 = *v40;
          if ( v24 == *v40 )
            goto LABEL_28;
          ++v82;
        }
        if ( !v58 )
          v58 = v40;
        ++v93;
        v57 = v95 + 1;
        if ( 4 * ((int)v95 + 1) < 3 * v30 )
        {
          if ( v30 - (v57 + HIDWORD(v95)) > v30 >> 3 )
            goto LABEL_69;
          v74 = v29;
          v83 = v26;
          sub_354B7C0((__int64)&v93, v30);
          if ( !v96 )
            goto LABEL_137;
          v29 = v74;
          v26 = v83;
          v60 = (v96 - 1) & v74;
          v57 = v95 + 1;
          v58 = (__int64 *)(v94 + 16LL * v60);
          v61 = *v58;
          if ( v24 == *v58 )
            goto LABEL_69;
          v84 = 1;
          v62 = 0;
          while ( v61 != -4096 )
          {
            if ( v61 == -8192 && !v62 )
              v62 = v58;
            v60 = (v96 - 1) & (v84 + v60);
            v58 = (__int64 *)(v94 + 16LL * v60);
            v61 = *v58;
            if ( v24 == *v58 )
              goto LABEL_69;
            ++v84;
          }
          goto LABEL_83;
        }
LABEL_67:
        v73 = v29;
        v81 = v26;
        sub_354B7C0((__int64)&v93, 2 * v30);
        if ( !v96 )
          goto LABEL_137;
        v29 = v73;
        v26 = v81;
        v56 = (v96 - 1) & v73;
        v57 = v95 + 1;
        v58 = (__int64 *)(v94 + 16LL * v56);
        v59 = *v58;
        if ( v24 == *v58 )
          goto LABEL_69;
        v87 = 1;
        v62 = 0;
        while ( v59 != -4096 )
        {
          if ( !v62 && v59 == -8192 )
            v62 = v58;
          v56 = (v96 - 1) & (v87 + v56);
          v58 = (__int64 *)(v94 + 16LL * v56);
          v59 = *v58;
          if ( v24 == *v58 )
            goto LABEL_69;
          ++v87;
        }
LABEL_83:
        if ( v62 )
          v58 = v62;
LABEL_69:
        LODWORD(v95) = v57;
        if ( *v58 != -4096 )
          --HIDWORD(v95);
        *v58 = v24;
        v42 = 0;
        v43 = (unsigned int *)(v58 + 1);
        *v43 = 0;
LABEL_29:
        v44 = *v38 + *(_DWORD *)(v27 + 20);
        if ( v44 <= v42 )
          goto LABEL_22;
        v27 += 32;
        *v43 = v44;
        if ( v28 == v27 )
          goto LABEL_31;
      }
    }
  }
LABEL_32:
  v91 = *v4;
  v92 = v4[v8 - 1];
  v45 = sub_35459D0(v78, v92);
  v46 = *(_QWORD *)v45;
  v47 = *(_QWORD *)v45 + 32LL * *(unsigned int *)(v45 + 8);
  if ( v47 != *(_QWORD *)v45 )
  {
    do
    {
      while ( 1 )
      {
        v48 = *(_QWORD *)(v46 + 8);
        if ( v91 == (v48 & 0xFFFFFFFFFFFFFFF8LL)
          && (((unsigned __int8)v48 ^ 6) & 6) == 0
          && (unsigned __int8)sub_3544720(a4, v46) )
        {
          break;
        }
        v46 += 32;
        if ( v46 == v47 )
          goto LABEL_41;
      }
      v90 = *(_DWORD *)sub_354B9A0((__int64)&v93, &v92) + 1;
      v49 = sub_354B9A0((__int64)&v93, &v91);
      v50 = v90;
      if ( *(_DWORD *)v49 >= v90 )
        v50 = *(_DWORD *)v49;
      v46 += 32;
      *(_DWORD *)sub_354B9A0((__int64)&v93, &v91) = v50;
    }
    while ( v46 != v47 );
  }
LABEL_41:
  v51 = sub_354B9A0((__int64)&v93, *(__int64 **)(a1 + 32));
  v52 = v94;
  v53 = 16LL * v96;
  *(_DWORD *)(a1 + 80) = *(_DWORD *)v51;
  return sub_C7D6A0(v52, v53, 8);
}
