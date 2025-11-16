// Function: sub_31AF800
// Address: 0x31af800
//
__int64 __fastcall sub_31AF800(__int64 *a1, __int64 *a2)
{
  __int64 v2; // rsi
  __int64 v3; // rdi
  __int64 v4; // r13
  __int64 result; // rax
  __int64 v6; // rdx
  __int64 v7; // r14
  unsigned int v8; // r12d
  __int64 v9; // rcx
  unsigned __int64 v10; // r8
  __int64 v11; // rsi
  __int64 v12; // r9
  __int64 *v13; // rdi
  _BYTE *v14; // r15
  __int64 v15; // rbx
  __int64 v16; // r15
  __int64 v17; // rax
  _QWORD *v18; // r8
  __int64 v19; // rdi
  _QWORD *v20; // rcx
  int v21; // edx
  unsigned int v22; // r12d
  __int64 *v23; // rbx
  __int64 v24; // rdi
  _QWORD *v25; // rdi
  _QWORD *v26; // rdx
  _QWORD *v27; // rax
  __int64 v28; // rax
  _QWORD *v29; // rcx
  int v30; // edx
  __int64 v31; // rdi
  int v32; // edx
  unsigned int v33; // r11d
  __int64 *v34; // rcx
  __int64 v35; // r10
  unsigned int v36; // edx
  __int64 *v37; // rcx
  __int64 v38; // rdi
  int v39; // edi
  _QWORD *v40; // rax
  _QWORD *v41; // rdx
  __int64 v42; // rax
  _QWORD *v43; // r9
  int v44; // edi
  unsigned int v45; // edx
  __int64 *v46; // rcx
  __int64 v47; // r9
  int v48; // edx
  __int64 v49; // r9
  int v50; // edx
  unsigned int v51; // r8d
  __int64 *v52; // rcx
  __int64 v53; // rdi
  int v54; // ecx
  __int64 v55; // r10
  unsigned int v56; // ebx
  int v57; // edi
  int v58; // r11d
  __int64 v59; // rdx
  __int64 v60; // rcx
  __int64 v61; // r8
  __int64 v62; // r9
  __int64 v63; // rax
  int v64; // ecx
  int v65; // r15d
  int v66; // ecx
  int v67; // r10d
  int v68; // ecx
  int v69; // r10d
  int v70; // r11d
  int v71; // esi
  int v72; // edi
  __int64 v73; // [rsp-1B8h] [rbp-1B8h] BYREF
  __int64 v74; // [rsp-1B0h] [rbp-1B0h]
  __int64 v75; // [rsp-1A8h] [rbp-1A8h]
  __int64 v76; // [rsp-1A0h] [rbp-1A0h]
  __int64 v77; // [rsp-198h] [rbp-198h]
  __int64 v78; // [rsp-190h] [rbp-190h]
  __int64 v79; // [rsp-188h] [rbp-188h]
  __int64 v80; // [rsp-180h] [rbp-180h]
  __int64 v81; // [rsp-178h] [rbp-178h]
  __int64 v82; // [rsp-170h] [rbp-170h]
  __int64 v83; // [rsp-168h] [rbp-168h]
  __int64 v84; // [rsp-160h] [rbp-160h]
  __int64 v85; // [rsp-158h] [rbp-158h] BYREF
  __int64 v86; // [rsp-150h] [rbp-150h]
  __int64 v87; // [rsp-148h] [rbp-148h]
  __int64 v88; // [rsp-140h] [rbp-140h]
  __int64 v89; // [rsp-138h] [rbp-138h]
  __int64 v90; // [rsp-130h] [rbp-130h]
  __int64 v91; // [rsp-128h] [rbp-128h]
  __int64 v92; // [rsp-120h] [rbp-120h]
  __int64 v93; // [rsp-118h] [rbp-118h]
  __int64 v94; // [rsp-110h] [rbp-110h]
  __int64 v95; // [rsp-108h] [rbp-108h]
  __int64 v96; // [rsp-100h] [rbp-100h]
  _QWORD v97[12]; // [rsp-F8h] [rbp-F8h] BYREF
  _QWORD v98[19]; // [rsp-98h] [rbp-98h] BYREF

  v2 = *a2;
  v3 = *a1;
  if ( *(_DWORD *)(*(_QWORD *)(v3 + 48) + 72LL) == 2 )
    return nullsub_2035(v3, v2);
  v4 = v3;
  result = *(unsigned int *)(v3 + 24);
  v6 = *(_QWORD *)(v3 + 8);
  if ( !(_DWORD)result )
    return result;
  v7 = v2;
  v8 = ((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4);
  v9 = (unsigned int)(result - 1);
  v10 = (unsigned int)v9 & v8;
  v11 = v6 + 16 * v10;
  v12 = *(_QWORD *)v11;
  v13 = (__int64 *)v11;
  if ( *(_QWORD *)v11 != v7 )
  {
    v55 = *(_QWORD *)v11;
    v56 = v10;
    v57 = 1;
    while ( v55 != -4096 )
    {
      v58 = v57 + 1;
      v56 = v9 & (v56 + v57);
      v13 = (__int64 *)(v6 + 16LL * v56);
      v55 = *v13;
      if ( *v13 == v7 )
        goto LABEL_6;
      v57 = v58;
    }
    return result;
  }
LABEL_6:
  result = v6 + 16 * result;
  if ( (__int64 *)result == v13 )
    return result;
  v14 = (_BYTE *)v13[1];
  if ( !v14 )
    return result;
  if ( v12 != v7 )
  {
    v71 = 1;
    while ( v12 != -4096 )
    {
      v72 = v71 + 1;
      v10 = (unsigned int)v9 & (v71 + (_DWORD)v10);
      v11 = v6 + 16LL * (unsigned int)v10;
      v12 = *(_QWORD *)v11;
      if ( *(_QWORD *)v11 == v7 )
        goto LABEL_9;
      v71 = v72;
    }
LABEL_2:
    BUG();
  }
LABEL_9:
  if ( result == v11 )
    goto LABEL_2;
  v15 = *(_QWORD *)(v11 + 8);
  if ( *(_DWORD *)(v15 + 16) == 1 )
  {
    v16 = sub_31B9B30(v4, *(_QWORD *)(v11 + 8), 0, 0);
    v17 = sub_31B9BF0(v4, v15, 0, 0);
    if ( v16 )
      *(_QWORD *)(v16 + 48) = v17;
    if ( v17 )
      *(_QWORD *)(v17 + 40) = v16;
    v11 = ((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4);
    while ( 1 )
    {
      v18 = *(_QWORD **)(v15 + 64);
      v19 = *(unsigned int *)(v15 + 80);
      v20 = &v18[v19];
      if ( !*(_DWORD *)(v15 + 72) || v18 == v20 )
        break;
      v40 = *(_QWORD **)(v15 + 64);
      while ( *v40 == -8192 || *v40 == -4096 )
      {
        if ( v20 == ++v40 )
          goto LABEL_17;
      }
      if ( v20 == v40 )
        break;
      v41 = *(_QWORD **)(v15 + 64);
      while ( 1 )
      {
        v42 = *v41;
        v43 = v41;
        if ( *v41 != -4096 && v42 != -8192 )
          break;
        if ( v20 == ++v41 )
        {
          v42 = v43[1];
          break;
        }
      }
      if ( (_DWORD)v19 )
      {
        v44 = v19 - 1;
        v45 = v44 & (((unsigned int)v42 >> 9) ^ ((unsigned int)v42 >> 4));
        v46 = &v18[v45];
        v47 = *v46;
        if ( v42 == *v46 )
        {
LABEL_57:
          *v46 = -8192;
          --*(_DWORD *)(v15 + 72);
          ++*(_DWORD *)(v15 + 76);
        }
        else
        {
          v68 = 1;
          while ( v47 != -4096 )
          {
            v69 = v68 + 1;
            v45 = v44 & (v68 + v45);
            v46 = &v18[v45];
            v47 = *v46;
            if ( v42 == *v46 )
              goto LABEL_57;
            v68 = v69;
          }
        }
      }
      v48 = *(_DWORD *)(v42 + 112);
      v49 = *(_QWORD *)(v42 + 96);
      if ( v48 )
      {
        v50 = v48 - 1;
        v51 = v50 & v11;
        v52 = (__int64 *)(v49 + 8LL * (v50 & (unsigned int)v11));
        v53 = *v52;
        if ( v15 == *v52 )
        {
LABEL_60:
          *v52 = -8192;
          --*(_DWORD *)(v42 + 104);
          ++*(_DWORD *)(v42 + 108);
        }
        else
        {
          v66 = 1;
          while ( v53 != -4096 )
          {
            v67 = v66 + 1;
            v51 = v50 & (v66 + v51);
            v52 = (__int64 *)(v49 + 8LL * v51);
            v53 = *v52;
            if ( v15 == *v52 )
              goto LABEL_60;
            v66 = v67;
          }
        }
      }
      if ( !*(_BYTE *)(v15 + 24) )
        --*(_DWORD *)(v42 + 20);
    }
LABEL_17:
    v10 = *(_QWORD *)(v15 + 96);
    v12 = *(unsigned int *)(v15 + 112);
    v21 = *(_DWORD *)(v15 + 104);
    while ( 1 )
    {
      if ( !v21 )
        goto LABEL_19;
      v25 = (_QWORD *)(v10 + 8LL * (unsigned int)v12);
      if ( v25 == (_QWORD *)v10 )
        goto LABEL_19;
      v26 = (_QWORD *)v10;
      v27 = (_QWORD *)v10;
      while ( *v27 == -8192 || *v27 == -4096 )
      {
        if ( v25 == ++v27 )
          goto LABEL_19;
      }
      if ( v25 == v27 )
        goto LABEL_19;
      while ( 1 )
      {
        v28 = *v26;
        v29 = v26;
        if ( *v26 != -4096 && v28 != -8192 )
          break;
        if ( v25 == ++v26 )
        {
          v28 = v29[1];
          break;
        }
      }
      v30 = *(_DWORD *)(v28 + 80);
      v31 = *(_QWORD *)(v28 + 64);
      if ( v30 )
      {
        v32 = v30 - 1;
        v33 = v32 & v11;
        v34 = (__int64 *)(v31 + 8LL * (v32 & (unsigned int)v11));
        v35 = *v34;
        if ( v15 == *v34 )
        {
LABEL_38:
          *v34 = -8192;
          --*(_DWORD *)(v28 + 72);
          ++*(_DWORD *)(v28 + 76);
          v10 = *(_QWORD *)(v15 + 96);
          v12 = *(unsigned int *)(v15 + 112);
        }
        else
        {
          v64 = 1;
          while ( v35 != -4096 )
          {
            v65 = v64 + 1;
            v33 = v32 & (v64 + v33);
            v34 = (__int64 *)(v31 + 8LL * v33);
            v35 = *v34;
            if ( v15 == *v34 )
              goto LABEL_38;
            v64 = v65;
          }
        }
      }
      if ( !(_DWORD)v12 )
        goto LABEL_65;
      v36 = (v12 - 1) & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
      v37 = (__int64 *)(v10 + 8LL * v36);
      v38 = *v37;
      if ( v28 != *v37 )
        break;
LABEL_41:
      *v37 = -8192;
      v39 = *(_DWORD *)(v15 + 104);
      ++*(_DWORD *)(v15 + 108);
      v10 = *(_QWORD *)(v15 + 96);
      v21 = v39 - 1;
      v12 = *(unsigned int *)(v15 + 112);
      *(_DWORD *)(v15 + 104) = v39 - 1;
LABEL_42:
      if ( !*(_BYTE *)(v28 + 24) )
        --*(_DWORD *)(v15 + 20);
    }
    v54 = 1;
    while ( v38 != -4096 )
    {
      v70 = v54 + 1;
      v36 = (v12 - 1) & (v54 + v36);
      v37 = (__int64 *)(v10 + 8LL * v36);
      v38 = *v37;
      if ( v28 == *v37 )
        goto LABEL_41;
      v54 = v70;
    }
LABEL_65:
    v21 = *(_DWORD *)(v15 + 104);
    goto LABEL_42;
  }
  if ( v14[24] )
    goto LABEL_21;
  (*(void (__fastcall **)(_QWORD *, _BYTE *, __int64))(*(_QWORD *)v14 + 24LL))(v98, v14, v4);
  (*(void (__fastcall **)(_QWORD *, _BYTE *, __int64))(*(_QWORD *)v14 + 16LL))(v97, v14, v4);
  v73 = v97[0];
  v74 = v97[1];
  v75 = v97[2];
  v76 = v97[3];
  v77 = v97[4];
  v78 = v97[5];
  v79 = v97[6];
  v80 = v97[7];
  v81 = v97[8];
  v82 = v97[9];
  v83 = v97[10];
  v84 = v97[11];
  v85 = v98[0];
  v86 = v98[1];
  v87 = v98[2];
  v88 = v98[3];
  v89 = v98[4];
  v90 = v98[5];
  v91 = v98[6];
  v92 = v98[7];
  v93 = v98[8];
  v94 = v98[9];
  v95 = v98[10];
  v96 = v98[11];
  while ( 1 )
  {
    v11 = (__int64)&v85;
    if ( (unsigned __int8)sub_31B8DE0(
                            &v73,
                            &v85,
                            v59,
                            v60,
                            v61,
                            v62,
                            v73,
                            v74,
                            v75,
                            v76,
                            v77,
                            v78,
                            v79,
                            v80,
                            v81,
                            v82,
                            v83,
                            v84,
                            v85,
                            v86,
                            v87,
                            v88,
                            v89,
                            v90,
                            v91,
                            v92,
                            v93,
                            v94,
                            v95,
                            v96) )
      break;
    v63 = sub_31B8B80(&v73);
    --*(_DWORD *)(v63 + 20);
    sub_31B8D10(&v73);
  }
LABEL_19:
  result = *(unsigned int *)(v4 + 24);
  v6 = *(_QWORD *)(v4 + 8);
  if ( (_DWORD)result )
  {
    v9 = (unsigned int)(result - 1);
LABEL_21:
    v22 = v9 & v8;
    v23 = (__int64 *)(v6 + 16LL * v22);
    result = *v23;
    if ( *v23 == v7 )
    {
LABEL_22:
      v24 = v23[1];
      if ( v24 )
        result = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, unsigned __int64, __int64))(*(_QWORD *)v24 + 8LL))(
                   v24,
                   v11,
                   v6,
                   v9,
                   v10,
                   v12);
      *v23 = -8192;
      --*(_DWORD *)(v4 + 16);
      ++*(_DWORD *)(v4 + 20);
    }
    else
    {
      v11 = 1;
      while ( result != -4096 )
      {
        v22 = v9 & (v11 + v22);
        v23 = (__int64 *)(v6 + 16LL * v22);
        result = *v23;
        if ( *v23 == v7 )
          goto LABEL_22;
        v11 = (unsigned int)(v11 + 1);
      }
    }
  }
  return result;
}
