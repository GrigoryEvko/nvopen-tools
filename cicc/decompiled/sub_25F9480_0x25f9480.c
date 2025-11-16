// Function: sub_25F9480
// Address: 0x25f9480
//
__int64 __fastcall sub_25F9480(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, char a5)
{
  int v5; // r13d
  __int64 result; // rax
  __int64 v9; // r12
  __int64 v10; // r8
  __int64 v11; // rdi
  __int64 v12; // r9
  __int64 v13; // rdi
  __int64 v14; // rax
  __int64 v15; // r8
  unsigned int v16; // esi
  __int64 *v17; // rdx
  __int64 v18; // r10
  __int64 v19; // rax
  __int64 v20; // r8
  unsigned int v21; // edx
  int *v22; // rcx
  int v23; // esi
  unsigned __int64 v24; // r8
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // r15
  __int64 v28; // rax
  __int64 v29; // rcx
  unsigned int v30; // r9d
  void *v31; // rax
  char *v32; // rax
  unsigned int v33; // esi
  char *v34; // rax
  __int64 v35; // r10
  __int64 v36; // rdx
  __int64 v37; // rsi
  int v38; // r8d
  unsigned int v39; // ecx
  int *v40; // r9
  int v41; // edi
  int v42; // edx
  int v43; // ecx
  int v44; // ecx
  int v45; // eax
  int v46; // ecx
  int v47; // r9d
  unsigned int v48; // r15d
  int v49; // r10d
  __int64 v50; // rdx
  __int64 v51; // r8
  unsigned int v52; // ecx
  int *v53; // rsi
  int v54; // edi
  int v55; // r9d
  int v56; // esi
  int v57; // r10d
  int v58; // r10d
  __int64 v60; // [rsp+8h] [rbp-68h]
  __int64 n; // [rsp+10h] [rbp-60h]
  size_t na; // [rsp+10h] [rbp-60h]
  int v63; // [rsp+18h] [rbp-58h]
  unsigned int v64; // [rsp+1Ch] [rbp-54h]
  unsigned __int64 v67; // [rsp+30h] [rbp-40h]
  __int64 v68; // [rsp+38h] [rbp-38h]

  result = *(_DWORD *)(a1 + 4) & 0x7FFFFFF;
  v60 = 8LL * (unsigned int)result;
  if ( (_DWORD)result )
  {
    v9 = 0;
    while ( 1 )
    {
      v26 = *(_QWORD *)(a1 - 8);
      v27 = *(_QWORD *)(v26 + 4 * v9);
      v68 = *(_QWORD *)(32LL * *(unsigned int *)(a1 + 72) + v26 + v9);
      if ( *(_BYTE *)v27 == 22 )
        break;
LABEL_19:
      sub_C7D6A0(0, 0, 8);
      v30 = *(_DWORD *)(a3 + 24);
      if ( v30 )
      {
        v63 = *(_DWORD *)(a3 + 24);
        n = 16LL * v30;
        v31 = (void *)sub_C7D670(n, 8);
        v32 = (char *)memcpy(v31, *(const void **)(a3 + 8), n);
        v10 = n;
        v11 = (__int64)v32;
        v33 = (v63 - 1) & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
        v34 = &v32[16 * v33];
        v35 = *(_QWORD *)v34;
        if ( v27 == *(_QWORD *)v34 )
        {
LABEL_21:
          if ( v34 != (char *)(v11 + n) )
            v27 = *((_QWORD *)v34 + 1);
        }
        else
        {
          v45 = 1;
          while ( v35 != -4096 )
          {
            v46 = v45 + 1;
            v33 = (v63 - 1) & (v45 + v33);
            v34 = (char *)(v11 + 16LL * v33);
            v35 = *(_QWORD *)v34;
            if ( v27 == *(_QWORD *)v34 )
              goto LABEL_21;
            v45 = v46;
          }
        }
      }
      else
      {
        v10 = 0;
        v11 = 0;
      }
      sub_C7D6A0(v11, v10, 8);
      v13 = *a2;
      v14 = *(unsigned int *)(*a2 + 48);
      v15 = *(_QWORD *)(*a2 + 32);
      if ( (_DWORD)v14 )
      {
        v12 = (unsigned int)(v14 - 1);
        v16 = v12 & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
        v17 = (__int64 *)(v15 + 16LL * v16);
        v18 = *v17;
        if ( v27 == *v17 )
        {
LABEL_6:
          if ( v17 != (__int64 *)(v15 + 16 * v14) )
            v5 = *((_DWORD *)v17 + 2);
        }
        else
        {
          v42 = 1;
          while ( v18 != -4096 )
          {
            v43 = v42 + 1;
            v16 = v12 & (v42 + v16);
            v17 = (__int64 *)(v15 + 16LL * v16);
            v18 = *v17;
            if ( v27 == *v17 )
              goto LABEL_6;
            v42 = v43;
          }
        }
      }
      v19 = *(unsigned int *)(v13 + 112);
      v20 = *(_QWORD *)(v13 + 96);
      if ( (_DWORD)v19 )
      {
        v21 = (v19 - 1) & (37 * v5);
        v22 = (int *)(v20 + 8LL * v21);
        v23 = *v22;
        if ( *v22 == v5 )
        {
LABEL_10:
          if ( v22 != (int *)(v20 + 8 * v19) )
            v64 = v22[1];
        }
        else
        {
          v44 = 1;
          while ( v23 != -1 )
          {
            v12 = (unsigned int)(v44 + 1);
            v21 = (v19 - 1) & (v44 + v21);
            v22 = (int *)(v20 + 8LL * v21);
            v23 = *v22;
            if ( *v22 == v5 )
              goto LABEL_10;
            v44 = v12;
          }
        }
      }
      v24 = v67 & 0xFFFFFFFF00000000LL | v64;
      v25 = *(unsigned int *)(a4 + 8);
      v67 = v24;
      if ( v25 + 1 > (unsigned __int64)*(unsigned int *)(a4 + 12) )
      {
        na = v24;
        sub_C8D5F0(a4, (const void *)(a4 + 16), v25 + 1, 0x10u, v24, v12);
        v25 = *(unsigned int *)(a4 + 8);
        v24 = na;
      }
      result = *(_QWORD *)a4 + 16 * v25;
      v9 += 8;
      *(_QWORD *)result = v24;
      *(_QWORD *)(result + 8) = v68;
      ++*(_DWORD *)(a4 + 8);
      if ( v60 == v9 )
        return result;
    }
    v28 = *(unsigned int *)(v27 + 32);
    if ( a5 )
    {
      v29 = a2[30];
LABEL_18:
      v27 = *(_QWORD *)(v29 + 32 * (v28 - (*(_DWORD *)(v29 + 4) & 0x7FFFFFF)));
      goto LABEL_19;
    }
    v36 = *((unsigned int *)a2 + 48);
    v37 = a2[22];
    if ( (_DWORD)v36 )
    {
      v38 = v36 - 1;
      v39 = (v36 - 1) & (37 * v28);
      v40 = (int *)(v37 + 16LL * v39);
      v41 = *v40;
      if ( (_DWORD)v28 == *v40 )
      {
LABEL_25:
        v27 = *((_QWORD *)v40 + 1);
        goto LABEL_19;
      }
      v47 = *v40;
      v48 = (v36 - 1) & (37 * v28);
      v49 = 1;
      while ( v47 != -1 )
      {
        v48 = v38 & (v49 + v48);
        v47 = *(_DWORD *)(v37 + 16LL * v48);
        if ( v47 == (_DWORD)v28 )
        {
          v55 = 1;
          while ( v41 != -1 )
          {
            v58 = v55 + 1;
            v39 = v38 & (v55 + v39);
            v40 = (int *)(v37 + 16LL * v39);
            v41 = *v40;
            if ( *v40 == (_DWORD)v28 )
              goto LABEL_25;
            v55 = v58;
          }
          v40 = (int *)(v37 + 16 * v36);
          goto LABEL_25;
        }
        ++v49;
      }
    }
    v50 = *((unsigned int *)a2 + 22);
    v51 = a2[9];
    if ( (_DWORD)v50 )
    {
      v52 = (v50 - 1) & (37 * v28);
      v53 = (int *)(v51 + 8LL * v52);
      v54 = *v53;
      if ( (_DWORD)v28 == *v53 )
      {
LABEL_43:
        v29 = a2[30];
        v28 = (unsigned int)v53[1];
        goto LABEL_18;
      }
      v56 = 1;
      while ( v54 != -1 )
      {
        v57 = v56 + 1;
        v52 = (v50 - 1) & (v56 + v52);
        v53 = (int *)(v51 + 8LL * v52);
        v54 = *v53;
        if ( (_DWORD)v28 == *v53 )
          goto LABEL_43;
        v56 = v57;
      }
    }
    v53 = (int *)(v51 + 8 * v50);
    goto LABEL_43;
  }
  return result;
}
