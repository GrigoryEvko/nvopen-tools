// Function: sub_FD7340
// Address: 0xfd7340
//
__int64 __fastcall sub_FD7340(__int64 a1, __int64 a2, __int64 a3, __int64 *a4, __int64 a5, __int64 a6)
{
  unsigned __int8 v9; // al
  __int64 v10; // rcx
  char v11; // al
  __int64 v12; // rax
  __int64 v13; // rdi
  unsigned __int64 *v14; // rdx
  _QWORD *v15; // rax
  char *v16; // r9
  char *v17; // r12
  __int64 v18; // r13
  unsigned __int64 v19; // rax
  _QWORD *v20; // r13
  _QWORD *v21; // r15
  _QWORD *v22; // r12
  __int64 v23; // rax
  const void *v24; // r13
  __int64 v25; // r12
  size_t v26; // r15
  __int64 result; // rax
  __int64 v28; // r15
  __int64 v29; // rax
  signed __int64 v30; // rax
  __int64 v31; // rdi
  __int64 v32; // rsi
  __int64 v33; // rdi
  __int64 v34; // rsi
  __int64 v35; // rdi
  __int64 v36; // rsi
  __int64 v37; // rdi
  __int64 v38; // r12
  __int64 v39; // rsi
  __int64 v40; // rax
  int v41; // eax
  int v42; // edx
  unsigned __int64 v43; // r13
  unsigned __int64 v44; // rsi
  bool v45; // cf
  unsigned __int64 v46; // r13
  unsigned __int64 *v47; // r15
  unsigned __int64 v48; // rsi
  unsigned __int64 v49; // rax
  unsigned __int64 *v50; // r12
  unsigned __int64 v51; // rax
  unsigned __int64 *v52; // r13
  unsigned __int64 v53; // rdx
  __int64 v54; // rdi
  __int64 v55; // r12
  __int64 v56; // rdi
  __int64 v57; // r12
  __int64 v58; // rdi
  __int64 v59; // r12
  __int64 v60; // r13
  __int64 v61; // rax
  unsigned __int64 *v62; // rcx
  unsigned __int64 *v63; // [rsp+0h] [rbp-60h]
  unsigned __int64 *v64; // [rsp+8h] [rbp-58h]
  __int64 v65; // [rsp+10h] [rbp-50h]
  char *v66; // [rsp+10h] [rbp-50h]
  unsigned __int64 *v67; // [rsp+10h] [rbp-50h]
  unsigned __int64 *v68; // [rsp+10h] [rbp-50h]
  char *v70; // [rsp+20h] [rbp-40h]
  __int64 v71; // [rsp+20h] [rbp-40h]
  _QWORD *v72; // [rsp+20h] [rbp-40h]
  char *v73; // [rsp+20h] [rbp-40h]
  unsigned __int64 *v74; // [rsp+20h] [rbp-40h]
  char *v75; // [rsp+20h] [rbp-40h]
  unsigned __int64 *v76; // [rsp+28h] [rbp-38h]
  unsigned __int64 *v77; // [rsp+28h] [rbp-38h]

  v9 = *(_BYTE *)(a1 + 67) & 0xCF | (*(_BYTE *)(a1 + 67) | *(_BYTE *)(a2 + 67)) & 0x30;
  *(_BYTE *)(a1 + 67) = v9;
  v10 = (v9 | *(_BYTE *)(a2 + 67)) & 0x40;
  v11 = v10 | v9 & 0xBF;
  *(_BYTE *)(a1 + 67) = v11;
  if ( (v11 & 0x40) == 0 )
  {
    v28 = *(_QWORD *)(a1 + 24);
    v29 = 48LL * *(unsigned int *)(a1 + 32);
    v10 = v28 + v29;
    v30 = 0xAAAAAAAAAAAAAAABLL * (v29 >> 4);
    v65 = v10;
    if ( v30 >> 2 )
    {
      v71 = v28 + 192 * (v30 >> 2);
      while ( 1 )
      {
        v37 = *(_QWORD *)(a2 + 24);
        v38 = v28;
        v39 = v37 + 48LL * *(unsigned int *)(a2 + 32);
        if ( v39 != sub_FD57F0(v37, v39, a4, v28) )
          break;
        v31 = *(_QWORD *)(a2 + 24);
        v28 += 48;
        v32 = v31 + 48LL * *(unsigned int *)(a2 + 32);
        if ( v32 != sub_FD57F0(v31, v32, a4, v28) )
          break;
        v33 = *(_QWORD *)(a2 + 24);
        v28 = v38 + 96;
        v34 = v33 + 48LL * *(unsigned int *)(a2 + 32);
        if ( v34 != sub_FD57F0(v33, v34, a4, v38 + 96) )
          break;
        v35 = *(_QWORD *)(a2 + 24);
        v28 = v38 + 144;
        v36 = v35 + 48LL * *(unsigned int *)(a2 + 32);
        if ( v36 != sub_FD57F0(v35, v36, a4, v38 + 144) )
          break;
        v28 = v38 + 192;
        if ( v71 == v38 + 192 )
        {
          v30 = 0xAAAAAAAAAAAAAAABLL * ((v65 - v28) >> 4);
          goto LABEL_73;
        }
      }
LABEL_34:
      if ( v65 != v28 )
        goto LABEL_2;
      goto LABEL_35;
    }
LABEL_73:
    if ( v30 != 2 )
    {
      if ( v30 != 3 )
      {
        if ( v30 != 1 )
          goto LABEL_35;
        goto LABEL_76;
      }
      v56 = *(_QWORD *)(a2 + 24);
      v57 = v56 + 48LL * *(unsigned int *)(a2 + 32);
      if ( v57 != sub_FD57F0(v56, v57, a4, v28) )
        goto LABEL_34;
      v28 += 48;
    }
    v58 = *(_QWORD *)(a2 + 24);
    v59 = v58 + 48LL * *(unsigned int *)(a2 + 32);
    if ( v59 != sub_FD57F0(v58, v59, a4, v28) )
      goto LABEL_34;
    v28 += 48;
LABEL_76:
    v54 = *(_QWORD *)(a2 + 24);
    v55 = v54 + 48LL * *(unsigned int *)(a2 + 32);
    if ( v55 != sub_FD57F0(v54, v55, a4, v28) )
      goto LABEL_34;
LABEL_35:
    *(_BYTE *)(a1 + 67) |= 0x40u;
  }
LABEL_2:
  v12 = *(unsigned int *)(a1 + 32);
  v13 = a1 + 24;
  if ( (_DWORD)v12 )
  {
    v24 = *(const void **)(a2 + 24);
    v25 = *(unsigned int *)(a2 + 32);
    v26 = 48 * v25;
    if ( v12 + v25 > (unsigned __int64)*(unsigned int *)(a1 + 36) )
    {
      sub_C8D5F0(v13, (const void *)(a1 + 40), v12 + v25, 0x30u, a5, a6);
      v12 = *(unsigned int *)(a1 + 32);
    }
    if ( v26 )
    {
      memcpy((void *)(*(_QWORD *)(a1 + 24) + 48 * v12), v24, v26);
      LODWORD(v12) = *(_DWORD *)(a1 + 32);
    }
    *(_DWORD *)(a1 + 32) = v25 + v12;
    *(_DWORD *)(a2 + 32) = 0;
  }
  else
  {
    sub_FD7100(v13, a2 + 24, a3, v10, a5, a6);
  }
  v14 = *(unsigned __int64 **)(a1 + 48);
  v15 = *(_QWORD **)(a1 + 40);
  v16 = *(char **)(a2 + 48);
  v17 = *(char **)(a2 + 40);
  if ( v14 != v15 )
  {
    if ( v16 != v17 )
    {
      v18 = v16 - v17;
      if ( *(_QWORD *)(a1 + 56) - (_QWORD)v14 >= (unsigned __int64)(v16 - v17) )
      {
        do
        {
          if ( v14 )
          {
            *v14 = 0;
            v14[1] = 0;
            v19 = *((_QWORD *)v17 + 2);
            v14[2] = v19;
            if ( v19 != 0 && v19 != -4096 && v19 != -8192 )
            {
              v70 = v16;
              v76 = v14;
              sub_BD6050(v14, *(_QWORD *)v17 & 0xFFFFFFFFFFFFFFF8LL);
              v16 = v70;
              v14 = v76;
            }
          }
          v17 += 24;
          v14 += 3;
        }
        while ( v16 != v17 );
        *(_QWORD *)(a1 + 48) += v18;
LABEL_13:
        v20 = *(_QWORD **)(a2 + 40);
        v21 = *(_QWORD **)(a2 + 48);
        if ( v20 != v21 )
        {
          v22 = *(_QWORD **)(a2 + 40);
          do
          {
            v23 = v22[2];
            if ( v23 != -4096 && v23 != 0 && v23 != -8192 )
              sub_BD60C0(v22);
            v22 += 3;
          }
          while ( v21 != v22 );
          *(_QWORD *)(a2 + 48) = v20;
        }
        goto LABEL_38;
      }
      v43 = 0xAAAAAAAAAAAAAAABLL * (v18 >> 3);
      v44 = 0xAAAAAAAAAAAAAAABLL * (v14 - v15);
      if ( v43 > 0x555555555555555LL - v44 )
        sub_4262D8((__int64)"vector::_M_range_insert");
      if ( v43 < v44 )
        v43 = 0xAAAAAAAAAAAAAAABLL * (v14 - v15);
      v45 = __CFADD__(v44, v43);
      v46 = v44 + v43;
      if ( v45 )
      {
        v60 = 0xFFFFFFFFFFFFFFFLL;
      }
      else
      {
        if ( !v46 )
        {
          v63 = 0;
          v77 = 0;
          goto LABEL_46;
        }
        if ( v46 > 0x555555555555555LL )
          v46 = 0x555555555555555LL;
        v60 = 3 * v46;
      }
      v68 = *(unsigned __int64 **)(a1 + 48);
      v75 = *(char **)(a2 + 48);
      v61 = sub_22077B0(v60 * 8);
      v14 = v68;
      v16 = v75;
      v62 = (unsigned __int64 *)v61;
      v77 = (unsigned __int64 *)v61;
      v15 = *(_QWORD **)(a1 + 40);
      v47 = v62;
      v63 = &v62[v60];
      if ( v68 == v15 )
      {
        do
        {
LABEL_52:
          if ( v47 )
          {
            *v47 = 0;
            v47[1] = 0;
            v49 = *((_QWORD *)v17 + 2);
            v47[2] = v49;
            if ( v49 != 0 && v49 != -4096 && v49 != -8192 )
            {
              v67 = v14;
              v73 = v16;
              sub_BD6050(v47, *(_QWORD *)v17 & 0xFFFFFFFFFFFFFFF8LL);
              v14 = v67;
              v16 = v73;
            }
          }
          v17 += 24;
          v47 += 3;
        }
        while ( v16 != v17 );
        v50 = *(unsigned __int64 **)(a1 + 48);
        if ( v14 != v50 )
        {
          do
          {
            *v47 = 0;
            v47[1] = 0;
            v51 = v14[2];
            v47[2] = v51;
            if ( v51 != 0 && v51 != -4096 && v51 != -8192 )
            {
              v74 = v14;
              sub_BD6050(v47, *v14 & 0xFFFFFFFFFFFFFFF8LL);
              v14 = v74;
            }
            v14 += 3;
            v47 += 3;
          }
          while ( v50 != v14 );
          v50 = *(unsigned __int64 **)(a1 + 48);
        }
        v52 = *(unsigned __int64 **)(a1 + 40);
        if ( v52 != v50 )
        {
          do
          {
            v53 = v52[2];
            if ( v53 != 0 && v53 != -4096 && v53 != -8192 )
              sub_BD60C0(v52);
            v52 += 3;
          }
          while ( v52 != v50 );
          v50 = *(unsigned __int64 **)(a1 + 40);
        }
        if ( v50 )
          j_j___libc_free_0(v50, *(_QWORD *)(a1 + 56) - (_QWORD)v50);
        *(_QWORD *)(a1 + 48) = v47;
        *(_QWORD *)(a1 + 40) = v77;
        *(_QWORD *)(a1 + 56) = v63;
        goto LABEL_13;
      }
LABEL_46:
      v47 = v77;
      do
      {
        if ( v47 )
        {
          *v47 = 0;
          v47[1] = 0;
          v48 = v15[2];
          v47[2] = v48;
          if ( v48 != 0 && v48 != -4096 && v48 != -8192 )
          {
            v64 = v14;
            v66 = v16;
            v72 = v15;
            sub_BD6050(v47, *v15 & 0xFFFFFFFFFFFFFFF8LL);
            v14 = v64;
            v16 = v66;
            v15 = v72;
          }
        }
        v15 += 3;
        v47 += 3;
      }
      while ( v14 != v15 );
      goto LABEL_52;
    }
LABEL_25:
    *(_QWORD *)(a2 + 16) = a1;
    result = (*(_DWORD *)(a1 + 64) + 1) & 0x7FFFFFF | *(_DWORD *)(a1 + 64) & 0xF8000000;
    *(_DWORD *)(a1 + 64) = result;
    return result;
  }
  if ( v16 == v17 )
    goto LABEL_25;
  *(_QWORD *)(a1 + 40) = v17;
  v40 = *(_QWORD *)(a1 + 56);
  *(_QWORD *)(a1 + 48) = *(_QWORD *)(a2 + 48);
  *(_QWORD *)(a1 + 56) = *(_QWORD *)(a2 + 56);
  *(_QWORD *)(a2 + 40) = v14;
  *(_QWORD *)(a2 + 48) = v14;
  *(_QWORD *)(a2 + 56) = v40;
  *(_DWORD *)(a1 + 64) = (*(_DWORD *)(a1 + 64) + 1) & 0x7FFFFFF | *(_DWORD *)(a1 + 64) & 0xF8000000;
LABEL_38:
  *(_QWORD *)(a2 + 16) = a1;
  *(_DWORD *)(a1 + 64) = (*(_DWORD *)(a1 + 64) + 1) & 0x7FFFFFF | *(_DWORD *)(a1 + 64) & 0xF8000000;
  v41 = *(_DWORD *)(a2 + 64);
  v42 = (v41 + 0x7FFFFFF) & 0x7FFFFFF;
  result = v42 | v41 & 0xF8000000;
  *(_DWORD *)(a2 + 64) = result;
  if ( !v42 )
    return sub_FD59A0(a2, a3);
  return result;
}
