// Function: sub_2905B80
// Address: 0x2905b80
//
__int64 *__fastcall sub_2905B80(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v6; // r12
  __int64 v7; // r13
  __int64 v8; // rax
  __int64 *v9; // r13
  __int64 *v10; // r15
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  __int64 v23; // rax
  int v24; // edx
  __int64 v25; // r8
  int v26; // ecx
  unsigned int v27; // edx
  __int64 *v28; // rdi
  __int64 v29; // r9
  __int64 v31; // rax
  int v32; // edx
  __int64 v33; // r8
  __int64 v34; // rsi
  int v35; // ecx
  unsigned int v36; // edx
  __int64 *v37; // rdi
  __int64 v38; // r9
  int v39; // edx
  __int64 v40; // r8
  __int64 v41; // rsi
  int v42; // ecx
  unsigned int v43; // edx
  __int64 v44; // r9
  int v45; // edi
  int v46; // r10d
  int v47; // edx
  __int64 v48; // r8
  __int64 v49; // rsi
  int v50; // ecx
  unsigned int v51; // edx
  __int64 v52; // r9
  int v53; // edi
  int v54; // r10d
  int v55; // edi
  int v56; // r10d
  int v57; // edi
  int v58; // r10d
  __int64 v59; // [rsp+0h] [rbp-50h] BYREF
  __int64 v60; // [rsp+8h] [rbp-48h]
  __int64 v61[7]; // [rsp+18h] [rbp-38h] BYREF

  v6 = a1;
  v7 = (a2 - (__int64)a1) >> 5;
  v8 = (a2 - (__int64)a1) >> 3;
  v59 = a3;
  v60 = a4;
  if ( v7 <= 0 )
  {
LABEL_29:
    if ( v8 != 2 )
    {
      if ( v8 != 3 )
      {
        if ( v8 != 1 )
          return (__int64 *)a2;
        goto LABEL_37;
      }
      if ( (unsigned __int8)sub_2905AC0(&v59, v6, a3, a4, a5, a6) )
        return v6;
      ++v6;
    }
    if ( (unsigned __int8)sub_2905AC0(&v59, v6, a3, a4, a5, a6) )
      return v6;
    ++v6;
LABEL_37:
    if ( !(unsigned __int8)sub_2905AC0(&v59, v6, a3, a4, a5, a6) )
      return (__int64 *)a2;
    return v6;
  }
  v9 = &a1[4 * v7];
  while ( 1 )
  {
    v61[0] = *v6;
    if ( **(_BYTE **)sub_1152A40(v59, v61, a3, a4, a5, a6) <= 0x15u )
      break;
    v10 = v6 + 1;
    v61[0] = v6[1];
    if ( **(_BYTE **)sub_1152A40(v59, v61, v19, v20, v21, v22) <= 0x15u )
    {
      v31 = v60;
      v32 = *(_DWORD *)(v60 + 24);
      v33 = *(_QWORD *)(v60 + 8);
      if ( v32 )
      {
        v34 = v6[1];
        v35 = v32 - 1;
        v36 = (v32 - 1) & (((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4));
        v37 = (__int64 *)(v33 + 8LL * v36);
        v38 = *v37;
        if ( v34 != *v37 )
        {
          v57 = 1;
          while ( v38 != -4096 )
          {
            v58 = v57 + 1;
            v36 = v35 & (v57 + v36);
            v37 = (__int64 *)(v33 + 8LL * v36);
            v38 = *v37;
            if ( v34 == *v37 )
              goto LABEL_14;
            v57 = v58;
          }
          return v10;
        }
        goto LABEL_14;
      }
      return v10;
    }
    v10 = v6 + 2;
    v61[0] = v6[2];
    if ( **(_BYTE **)sub_1152A40(v59, v61, v11, v12, v13, v14) <= 0x15u )
    {
      v31 = v60;
      v39 = *(_DWORD *)(v60 + 24);
      v40 = *(_QWORD *)(v60 + 8);
      if ( v39 )
      {
        v41 = v6[2];
        v42 = v39 - 1;
        v43 = (v39 - 1) & (((unsigned int)v41 >> 9) ^ ((unsigned int)v41 >> 4));
        v37 = (__int64 *)(v40 + 8LL * v43);
        v44 = *v37;
        if ( *v37 != v41 )
        {
          v45 = 1;
          while ( v44 != -4096 )
          {
            v46 = v45 + 1;
            v43 = v42 & (v45 + v43);
            v37 = (__int64 *)(v40 + 8LL * v43);
            v44 = *v37;
            if ( v41 == *v37 )
              goto LABEL_14;
            v45 = v46;
          }
          return v10;
        }
        goto LABEL_14;
      }
      return v10;
    }
    v10 = v6 + 3;
    v61[0] = v6[3];
    if ( **(_BYTE **)sub_1152A40(v59, v61, v15, v16, v17, v18) <= 0x15u )
    {
      v31 = v60;
      v47 = *(_DWORD *)(v60 + 24);
      v48 = *(_QWORD *)(v60 + 8);
      if ( v47 )
      {
        v49 = v6[3];
        v50 = v47 - 1;
        v51 = (v47 - 1) & (((unsigned int)v49 >> 9) ^ ((unsigned int)v49 >> 4));
        v37 = (__int64 *)(v48 + 8LL * v51);
        v52 = *v37;
        if ( *v37 != v49 )
        {
          v53 = 1;
          while ( v52 != -4096 )
          {
            v54 = v53 + 1;
            v51 = v50 & (v53 + v51);
            v37 = (__int64 *)(v48 + 8LL * v51);
            v52 = *v37;
            if ( v49 == *v37 )
              goto LABEL_14;
            v53 = v54;
          }
          return v10;
        }
LABEL_14:
        *v37 = -8192;
        --*(_DWORD *)(v31 + 16);
        ++*(_DWORD *)(v31 + 20);
      }
      return v10;
    }
    v6 += 4;
    if ( v6 == v9 )
    {
      v8 = (a2 - (__int64)v6) >> 3;
      goto LABEL_29;
    }
  }
  v23 = v60;
  v24 = *(_DWORD *)(v60 + 24);
  v25 = *(_QWORD *)(v60 + 8);
  if ( v24 )
  {
    v26 = v24 - 1;
    v27 = (v24 - 1) & (((unsigned int)*v6 >> 9) ^ ((unsigned int)*v6 >> 4));
    v28 = (__int64 *)(v25 + 8LL * v27);
    v29 = *v28;
    if ( *v6 == *v28 )
    {
LABEL_10:
      *v28 = -8192;
      --*(_DWORD *)(v23 + 16);
      ++*(_DWORD *)(v23 + 20);
    }
    else
    {
      v55 = 1;
      while ( v29 != -4096 )
      {
        v56 = v55 + 1;
        v27 = v26 & (v55 + v27);
        v28 = (__int64 *)(v25 + 8LL * v27);
        v29 = *v28;
        if ( *v6 == *v28 )
          goto LABEL_10;
        v55 = v56;
      }
    }
  }
  return v6;
}
