// Function: sub_1E2FBD0
// Address: 0x1e2fbd0
//
_QWORD *__fastcall sub_1E2FBD0(__int64 a1, __int64 a2)
{
  __int64 v2; // r8
  unsigned int v5; // esi
  __int64 v6; // rdi
  unsigned int v7; // ecx
  __int64 *v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r12
  unsigned int v11; // edx
  __int64 v12; // r14
  __int64 v13; // rcx
  int v14; // eax
  int v15; // edx
  __int64 v16; // rsi
  int v17; // edi
  unsigned int v18; // eax
  __int64 *v19; // r15
  __int64 v20; // rcx
  __int64 v21; // rax
  unsigned __int64 *v22; // rax
  unsigned __int64 *v23; // r13
  _QWORD *v24; // r14
  __int64 v25; // rax
  __int64 v26; // rax
  bool v27; // zf
  __int64 v28; // rdx
  bool v29; // al
  unsigned __int64 v30; // rdi
  _QWORD *result; // rax
  __int64 *v32; // r15
  __int64 *v33; // r13
  unsigned int v34; // esi
  __int64 v35; // r8
  unsigned int v36; // ecx
  __int64 v37; // rdx
  _BYTE *v38; // rsi
  _QWORD *v39; // r14
  __int64 v40; // rcx
  int v41; // edx
  int v42; // edx
  __int64 v43; // r8
  unsigned int v44; // esi
  int v45; // eax
  _QWORD *v46; // rdi
  __int64 v47; // rcx
  int v48; // r10d
  _QWORD *v49; // r9
  int v50; // r10d
  int v51; // eax
  _QWORD *v52; // rdi
  int v53; // edx
  int v54; // edx
  __int64 v55; // r8
  int v56; // r10d
  unsigned int v57; // esi
  __int64 v58; // rcx
  int v59; // r14d
  __int64 *v60; // r11
  int v61; // edi
  int v62; // ecx
  int v63; // eax
  int v64; // esi
  __int64 v65; // r8
  unsigned int v66; // edx
  __int64 v67; // rdi
  int v68; // r10d
  __int64 *v69; // r9
  int v70; // eax
  int v71; // edx
  __int64 v72; // rdi
  int v73; // r9d
  unsigned int v74; // r12d
  __int64 *v75; // r8
  __int64 v76; // rsi
  __int64 *v77; // r10
  __int64 v78; // [rsp+8h] [rbp-98h]
  unsigned __int64 *v79; // [rsp+10h] [rbp-90h]
  char v80; // [rsp+1Bh] [rbp-85h]
  unsigned int v81; // [rsp+1Ch] [rbp-84h]
  __int64 v82; // [rsp+20h] [rbp-80h] BYREF
  __int64 v83; // [rsp+28h] [rbp-78h] BYREF
  unsigned int v84; // [rsp+30h] [rbp-70h]
  void *v85; // [rsp+40h] [rbp-60h] BYREF
  _QWORD v86[2]; // [rsp+48h] [rbp-58h] BYREF
  __int64 v87; // [rsp+58h] [rbp-48h]
  __int64 v88; // [rsp+60h] [rbp-40h]

  v2 = a1 + 8;
  v5 = *(_DWORD *)(a1 + 32);
  if ( !v5 )
  {
    ++*(_QWORD *)(a1 + 8);
    goto LABEL_89;
  }
  v6 = *(_QWORD *)(a1 + 16);
  v7 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v8 = (__int64 *)(v6 + 32LL * v7);
  v9 = *v8;
  if ( a2 == *v8 )
  {
    v10 = v8[2];
    v11 = *((_DWORD *)v8 + 6);
    v12 = 40LL * v11;
    goto LABEL_4;
  }
  v59 = 1;
  v60 = 0;
  while ( v9 != -8 )
  {
    if ( v60 || v9 != -16 )
      v8 = v60;
    v7 = (v5 - 1) & (v59 + v7);
    v77 = (__int64 *)(v6 + 32LL * v7);
    v9 = *v77;
    if ( a2 == *v77 )
    {
      v10 = v77[2];
      v11 = *((_DWORD *)v77 + 6);
      v8 = (__int64 *)(v6 + 32LL * v7);
      v12 = 40LL * v11;
      goto LABEL_4;
    }
    ++v59;
    v60 = v8;
    v8 = (__int64 *)(v6 + 32LL * v7);
  }
  v61 = *(_DWORD *)(a1 + 24);
  if ( v60 )
    v8 = v60;
  ++*(_QWORD *)(a1 + 8);
  v62 = v61 + 1;
  if ( 4 * (v61 + 1) >= 3 * v5 )
  {
LABEL_89:
    sub_1E2E3C0(v2, 2 * v5);
    v63 = *(_DWORD *)(a1 + 32);
    if ( v63 )
    {
      v64 = v63 - 1;
      v65 = *(_QWORD *)(a1 + 16);
      v66 = (v63 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v62 = *(_DWORD *)(a1 + 24) + 1;
      v8 = (__int64 *)(v65 + 32LL * v66);
      v67 = *v8;
      if ( a2 != *v8 )
      {
        v68 = 1;
        v69 = 0;
        while ( v67 != -8 )
        {
          if ( !v69 && v67 == -16 )
            v69 = v8;
          v66 = v64 & (v68 + v66);
          v8 = (__int64 *)(v65 + 32LL * v66);
          v67 = *v8;
          if ( a2 == *v8 )
            goto LABEL_80;
          ++v68;
        }
        if ( v69 )
          v8 = v69;
      }
      goto LABEL_80;
    }
    goto LABEL_124;
  }
  if ( v5 - *(_DWORD *)(a1 + 28) - v62 <= v5 >> 3 )
  {
    sub_1E2E3C0(v2, v5);
    v70 = *(_DWORD *)(a1 + 32);
    if ( v70 )
    {
      v71 = v70 - 1;
      v72 = *(_QWORD *)(a1 + 16);
      v73 = 1;
      v74 = (v70 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v75 = 0;
      v62 = *(_DWORD *)(a1 + 24) + 1;
      v8 = (__int64 *)(v72 + 32LL * v74);
      v76 = *v8;
      if ( a2 != *v8 )
      {
        while ( v76 != -8 )
        {
          if ( v76 == -16 && !v75 )
            v75 = v8;
          v74 = v71 & (v73 + v74);
          v8 = (__int64 *)(v72 + 32LL * v74);
          v76 = *v8;
          if ( a2 == *v8 )
            goto LABEL_80;
          ++v73;
        }
        if ( v75 )
          v8 = v75;
      }
      goto LABEL_80;
    }
LABEL_124:
    ++*(_DWORD *)(a1 + 24);
    BUG();
  }
LABEL_80:
  *(_DWORD *)(a1 + 24) = v62;
  if ( *v8 != -8 )
    --*(_DWORD *)(a1 + 28);
  *v8 = a2;
  v12 = 0;
  v11 = 0;
  v10 = 0;
  v8[1] = 0;
  v8[2] = 0;
  *((_DWORD *)v8 + 6) = 0;
LABEL_4:
  v13 = v8[1];
  v8[1] = 0;
  v14 = *(_DWORD *)(a1 + 32);
  v83 = v10;
  v82 = v13;
  v84 = v11;
  if ( v14 )
  {
    v15 = v14 - 1;
    v16 = *(_QWORD *)(a1 + 16);
    v17 = 1;
    v18 = (v14 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v19 = (__int64 *)(v16 + 32LL * v18);
    v20 = *v19;
    if ( a2 == *v19 )
    {
LABEL_6:
      v21 = v19[1];
      if ( (v21 & 4) != 0 )
      {
        v22 = (unsigned __int64 *)(v21 & 0xFFFFFFFFFFFFFFF8LL);
        v23 = v22;
        if ( v22 )
        {
          if ( (unsigned __int64 *)*v22 != v22 + 2 )
            _libc_free(*v22);
          j_j___libc_free_0(v23, 48);
        }
      }
      *v19 = -16;
      --*(_DWORD *)(a1 + 24);
      ++*(_DWORD *)(a1 + 28);
    }
    else
    {
      while ( v20 != -8 )
      {
        v18 = v15 & (v17 + v18);
        v19 = (__int64 *)(v16 + 32LL * v18);
        v20 = *v19;
        if ( a2 == *v19 )
          goto LABEL_6;
        ++v17;
      }
    }
  }
  v24 = (_QWORD *)(*(_QWORD *)(a1 + 40) + v12);
  v86[0] = 2;
  v86[1] = 0;
  v87 = 0;
  v85 = &unk_49FBDD8;
  v88 = 0;
  v25 = v24[3];
  if ( v25 )
  {
    if ( v25 == -8 || v25 == -16 )
    {
      v24[3] = 0;
      v29 = v87 != 0 && v87 != -8 && v87 != -16;
      v28 = 0;
LABEL_17:
      v24[4] = v28;
      v85 = &unk_49EE2B0;
      if ( v29 )
        sub_1649B30(v86);
      goto LABEL_19;
    }
    sub_1649B30(v24 + 1);
    v26 = v87;
    v27 = v87 == -8;
    v24[3] = v87;
    if ( v26 != 0 && !v27 )
    {
      sub_1649AC0(v24 + 1, v86[0] & 0xFFFFFFFFFFFFFFF8LL);
      v28 = v88;
      v29 = v87 != 0 && v87 != -16 && v87 != -8;
      goto LABEL_17;
    }
    v24[4] = v88;
  }
  else
  {
    v24[4] = 0;
  }
LABEL_19:
  v30 = v82 & 0xFFFFFFFFFFFFFFF8LL;
  result = (_QWORD *)((v82 >> 2) & 1);
  v79 = (unsigned __int64 *)(v82 & 0xFFFFFFFFFFFFFFF8LL);
  v80 = (v82 >> 2) & 1;
  if ( ((v82 >> 2) & 1) != 0 )
  {
    v32 = *(__int64 **)v30;
    v33 = (__int64 *)(*(_QWORD *)v30 + 8LL * *(unsigned int *)(v30 + 8));
    if ( *(__int64 **)v30 == v33 )
    {
LABEL_32:
      result = v79;
      if ( v79 )
      {
        if ( (unsigned __int64 *)*v79 != v79 + 2 )
          _libc_free(*v79);
        return (_QWORD *)j_j___libc_free_0(v79, 48);
      }
      return result;
    }
  }
  else
  {
    if ( !v79 )
      return result;
    v32 = &v82;
    v33 = &v83;
  }
  result = (_QWORD *)(a1 + 64);
  v81 = ((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4);
  v78 = a1 + 64;
  do
  {
    v39 = (_QWORD *)*v32;
    v85 = v39;
    if ( (*v39 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
      break;
    if ( (*((_BYTE *)v39 + 9) & 0xC) == 8 )
    {
      *((_BYTE *)v39 + 8) |= 4u;
      v40 = sub_38CE440(v39[3]);
      result = (_QWORD *)(v40 | *v39 & 7LL);
      *v39 = result;
      if ( v40 )
        break;
      v34 = *(_DWORD *)(a1 + 88);
      if ( !v34 )
      {
LABEL_39:
        ++*(_QWORD *)(a1 + 64);
        goto LABEL_40;
      }
    }
    else
    {
      v34 = *(_DWORD *)(a1 + 88);
      if ( !v34 )
        goto LABEL_39;
    }
    v35 = *(_QWORD *)(a1 + 72);
    v36 = (v34 - 1) & v81;
    result = (_QWORD *)(v35 + 32LL * v36);
    v37 = *result;
    if ( v10 != *result )
    {
      v50 = 1;
      v46 = 0;
      while ( v37 != -8 )
      {
        if ( v37 == -16 && !v46 )
          v46 = result;
        v36 = (v34 - 1) & (v50 + v36);
        result = (_QWORD *)(v35 + 32LL * v36);
        v37 = *result;
        if ( v10 == *result )
          goto LABEL_25;
        ++v50;
      }
      if ( !v46 )
        v46 = result;
      v51 = *(_DWORD *)(a1 + 80);
      ++*(_QWORD *)(a1 + 64);
      v45 = v51 + 1;
      if ( 4 * v45 >= 3 * v34 )
      {
LABEL_40:
        sub_1E2F9B0(v78, 2 * v34);
        v41 = *(_DWORD *)(a1 + 88);
        if ( !v41 )
          goto LABEL_125;
        v42 = v41 - 1;
        v43 = *(_QWORD *)(a1 + 72);
        v44 = v42 & v81;
        v45 = *(_DWORD *)(a1 + 80) + 1;
        v46 = (_QWORD *)(v43 + 32LL * (v42 & v81));
        v47 = *v46;
        if ( *v46 != v10 )
        {
          v48 = 1;
          v49 = 0;
          while ( v47 != -8 )
          {
            if ( !v49 && v47 == -16 )
              v49 = v46;
            v44 = v42 & (v48 + v44);
            v46 = (_QWORD *)(v43 + 32LL * v44);
            v47 = *v46;
            if ( v10 == *v46 )
              goto LABEL_53;
            ++v48;
          }
LABEL_44:
          if ( v49 )
            v46 = v49;
        }
      }
      else if ( v34 - *(_DWORD *)(a1 + 84) - v45 <= v34 >> 3 )
      {
        sub_1E2F9B0(v78, v34);
        v53 = *(_DWORD *)(a1 + 88);
        if ( !v53 )
        {
LABEL_125:
          ++*(_DWORD *)(a1 + 80);
          BUG();
        }
        v54 = v53 - 1;
        v55 = *(_QWORD *)(a1 + 72);
        v49 = 0;
        v56 = 1;
        v57 = v54 & v81;
        v45 = *(_DWORD *)(a1 + 80) + 1;
        v46 = (_QWORD *)(v55 + 32LL * (v54 & v81));
        v58 = *v46;
        if ( *v46 != v10 )
        {
          while ( v58 != -8 )
          {
            if ( v58 == -16 && !v49 )
              v49 = v46;
            v57 = v54 & (v56 + v57);
            v46 = (_QWORD *)(v55 + 32LL * v57);
            v58 = *v46;
            if ( v10 == *v46 )
              goto LABEL_53;
            ++v56;
          }
          goto LABEL_44;
        }
      }
LABEL_53:
      *(_DWORD *)(a1 + 80) = v45;
      if ( *v46 != -8 )
        --*(_DWORD *)(a1 + 84);
      *v46 = v10;
      v38 = 0;
      v52 = v46 + 1;
      *v52 = 0;
      v52[1] = 0;
      v52[2] = 0;
      goto LABEL_56;
    }
LABEL_25:
    v38 = (_BYTE *)result[2];
    if ( v38 != (_BYTE *)result[3] )
    {
      if ( v38 )
      {
        *(_QWORD *)v38 = v85;
        v38 = (_BYTE *)result[2];
      }
      result[2] = v38 + 8;
      goto LABEL_29;
    }
    v52 = result + 1;
LABEL_56:
    result = sub_1E2D470((__int64)v52, v38, &v85);
LABEL_29:
    ++v32;
  }
  while ( v33 != v32 );
  if ( v80 )
    goto LABEL_32;
  return result;
}
