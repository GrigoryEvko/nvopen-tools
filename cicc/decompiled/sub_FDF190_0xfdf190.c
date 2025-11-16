// Function: sub_FDF190
// Address: 0xfdf190
//
unsigned __int64 __fastcall sub_FDF190(_QWORD *a1)
{
  unsigned __int64 result; // rax
  unsigned __int64 *v3; // rbx
  __int64 v4; // r8
  __int64 v5; // r9
  __int64 v6; // rcx
  unsigned __int64 v7; // rdx
  __int64 *v8; // r12
  __int64 *v9; // rbx
  __int64 v10; // r13
  unsigned __int64 *v11; // r15
  _QWORD *v12; // rax
  __int64 v13; // rdx
  __int64 *i; // rcx
  __int64 **v15; // r13
  __int64 v16; // r12
  __int64 v17; // rbx
  __int64 v18; // rdi
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 *v21; // rbx
  __int64 *v22; // r13
  __int64 v23; // r15
  __int64 v24; // r12
  unsigned __int64 *v25; // rdx
  _QWORD *v26; // rax
  __int64 v27; // rdx
  __int64 v28; // r15
  unsigned __int64 v29; // rbx
  __int64 v30; // rax
  __int64 v31; // rcx
  __int64 v32; // rsi
  int v33; // eax
  int v34; // edi
  unsigned int v35; // edx
  __int64 *v36; // rax
  __int64 v37; // rax
  unsigned int v38; // eax
  __int64 v39; // rdx
  __int64 v40; // r13
  __int64 v41; // rax
  __int64 v42; // rdx
  __int64 **v43; // r13
  __int64 v44; // rax
  __int64 *v45; // rdi
  bool v46; // al
  __int64 *v47; // r12
  __int64 v48; // rax
  __int64 v49; // rax
  __int64 v50; // rdi
  unsigned __int64 *v51; // rbx
  unsigned __int64 *v52; // r12
  __int64 v53; // rdi
  __int64 v54; // rcx
  int v55; // eax
  int v56; // r10d
  __int64 v57; // [rsp-90h] [rbp-90h]
  __int64 v58; // [rsp-88h] [rbp-88h] BYREF
  __int64 v59; // [rsp-80h] [rbp-80h]
  __int64 *v60; // [rsp-78h] [rbp-78h]
  unsigned __int64 v61; // [rsp-70h] [rbp-70h]
  __int64 v62; // [rsp-68h] [rbp-68h]
  unsigned __int64 *v63; // [rsp-60h] [rbp-60h]
  _QWORD *v64; // [rsp-58h] [rbp-58h]
  unsigned __int64 v65; // [rsp-50h] [rbp-50h]
  __int64 v66; // [rsp-48h] [rbp-48h]
  unsigned __int64 *v67; // [rsp-40h] [rbp-40h]

  result = a1[15];
  if ( *(_QWORD *)(result + 32) == *(_QWORD *)(result + 40) )
    return result;
  v60 = 0;
  v61 = 0;
  v62 = 0;
  v63 = 0;
  v64 = 0;
  v65 = 0;
  v66 = 0;
  v67 = 0;
  v59 = 8;
  v58 = sub_22077B0(64);
  v3 = (unsigned __int64 *)(v58 + 24);
  result = sub_22077B0(512);
  v6 = a1[15];
  v63 = (unsigned __int64 *)(v58 + 24);
  v7 = result + 512;
  *(_QWORD *)(v58 + 24) = result;
  v67 = v3;
  v65 = result;
  v66 = result + 512;
  v64 = (_QWORD *)result;
  v8 = *(__int64 **)(v6 + 40);
  v62 = result + 512;
  v9 = *(__int64 **)(v6 + 32);
  v61 = result;
  v60 = (__int64 *)result;
  if ( v9 == v8 )
    goto LABEL_31;
  while ( 1 )
  {
    v10 = *v9;
    if ( result == v7 - 16 )
      break;
    if ( result )
    {
      *(_QWORD *)result = v10;
      *(_QWORD *)(result + 8) = 0;
      result = (unsigned __int64)v64;
    }
    result += 16LL;
    ++v9;
    v64 = (_QWORD *)result;
    if ( v8 == v9 )
      goto LABEL_15;
LABEL_7:
    v7 = v66;
  }
  v11 = v67;
  if ( 32 * (v67 - v63 - 1) + ((__int64)(result - v65) >> 4) + ((v62 - (__int64)v60) >> 4) == 0x7FFFFFFFFFFFFFFLL )
LABEL_64:
    sub_4262D8((__int64)"cannot create std::deque larger than max_size()");
  if ( (unsigned __int64)(v59 - (((__int64)v67 - v58) >> 3)) <= 1 )
  {
    sub_FDF010(&v58, 1u, 0);
    v11 = v67;
  }
  v11[1] = sub_22077B0(512);
  v12 = v64;
  if ( v64 )
  {
    *v64 = v10;
    v12[1] = 0;
  }
  ++v9;
  result = *++v67;
  v13 = *v67 + 512;
  v65 = result;
  v66 = v13;
  v64 = (_QWORD *)result;
  if ( v8 != v9 )
    goto LABEL_7;
LABEL_15:
  for ( i = v60; v60 != (__int64 *)result; i = v60 )
  {
    v15 = (__int64 **)*i;
    v16 = i[1];
    if ( i == (__int64 *)(v62 - 16) )
    {
      j_j___libc_free_0(v61, 512);
      v54 = *++v63 + 512;
      v61 = *v63;
      v62 = v54;
      v60 = (__int64 *)v61;
    }
    else
    {
      v60 += 2;
    }
    v17 = (unsigned int)sub_FDD0F0((__int64)a1, *v15[4]);
    v18 = sub_22077B0(192);
    *(_QWORD *)(v18 + 16) = v16;
    *(_BYTE *)(v18 + 24) = 0;
    *(_DWORD *)(v18 + 28) = 1;
    *(_QWORD *)(v18 + 32) = v18 + 48;
    *(_QWORD *)(v18 + 40) = 0x400000000LL;
    *(_QWORD *)(v18 + 112) = v18 + 128;
    *(_QWORD *)(v18 + 120) = 0x400000001LL;
    *(_QWORD *)(v18 + 144) = v18 + 160;
    *(_QWORD *)(v18 + 152) = 0x100000001LL;
    *(_DWORD *)(v18 + 128) = v17;
    *(_WORD *)(v18 + 184) = 0;
    *(_QWORD *)(v18 + 160) = 0;
    *(_QWORD *)(v18 + 168) = 0;
    *(_QWORD *)(v18 + 176) = 0;
    sub_2208C80(v18, a1 + 11);
    v19 = a1[8];
    v20 = a1[12];
    ++a1[13];
    *(_QWORD *)(v19 + 24 * v17 + 8) = v20 + 16;
    v21 = v15[1];
    v22 = v15[2];
    for ( result = (unsigned __int64)v64; v22 != v21; v64 = (_QWORD *)result )
    {
      while ( 1 )
      {
        v23 = *v21;
        v24 = a1[12] + 16LL;
        if ( result == v66 - 16 )
          break;
        if ( result )
        {
          *(_QWORD *)result = v23;
          *(_QWORD *)(result + 8) = v24;
          result = (unsigned __int64)v64;
        }
        result += 16LL;
        ++v21;
        v64 = (_QWORD *)result;
        if ( v22 == v21 )
          goto LABEL_30;
      }
      v25 = v67;
      if ( 32 * (v67 - v63 - 1) + ((__int64)(result - v65) >> 4) + ((v62 - (__int64)v60) >> 4) == 0x7FFFFFFFFFFFFFFLL )
        goto LABEL_64;
      if ( (unsigned __int64)(v59 - (((__int64)v67 - v58) >> 3)) <= 1 )
      {
        sub_FDF010(&v58, 1u, 0);
        v25 = v67;
      }
      v25[1] = sub_22077B0(512);
      v26 = v64;
      if ( v64 )
      {
        *v64 = v23;
        v26[1] = v24;
      }
      ++v21;
      result = *++v67;
      v27 = *v67 + 512;
      v65 = result;
      v66 = v27;
    }
LABEL_30:
    ;
  }
LABEL_31:
  v28 = a1[17];
  v29 = 0;
  if ( v28 != a1[18] )
  {
    while ( 1 )
    {
      v42 = 24 * v29 + a1[8];
      v43 = *(__int64 ***)(v42 + 8);
      if ( !v43 )
        break;
      v44 = *((unsigned int *)v43 + 3);
      v45 = v43[12];
      if ( (unsigned int)v44 <= 1 )
      {
        if ( *(_DWORD *)v42 != *(_DWORD *)v45 )
          break;
      }
      else
      {
        v57 = 24 * v29 + a1[8];
        v46 = sub_FDC990(v45, (_DWORD *)v45 + v44, (_DWORD *)v42);
        v42 = v57;
        if ( !v46 )
          break;
      }
      v47 = *v43;
      if ( *v43 )
      {
        v48 = *((unsigned int *)v47 + 3);
        if ( (unsigned int)v48 <= 1
          || !sub_FDC990((_DWORD *)v47[12], (_DWORD *)(v47[12] + 4 * v48), (_DWORD *)v42)
          || (v47 = (__int64 *)*v47) != 0 )
        {
          v49 = *((unsigned int *)v47 + 26);
          if ( v49 + 1 > (unsigned __int64)*((unsigned int *)v47 + 27) )
          {
            sub_C8D5F0((__int64)(v47 + 12), v47 + 14, v49 + 1, 4u, v4, v5);
            v49 = *((unsigned int *)v47 + 26);
          }
          *(_DWORD *)(v47[12] + 4 * v49) = v29;
          ++*((_DWORD *)v47 + 26);
        }
      }
LABEL_40:
      v28 = a1[17];
      ++v29;
      result = (a1[18] - v28) >> 3;
      if ( v29 >= result )
        goto LABEL_49;
    }
    v30 = a1[15];
    v31 = *(_QWORD *)(v28 + 8 * v29);
    v32 = *(_QWORD *)(v30 + 8);
    v33 = *(_DWORD *)(v30 + 24);
    if ( v33 )
    {
      v34 = v33 - 1;
      v35 = (v33 - 1) & (((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4));
      v36 = (__int64 *)(v32 + 16LL * v35);
      v5 = *v36;
      if ( v31 == *v36 )
      {
LABEL_36:
        v37 = v36[1];
        if ( v37 )
        {
          v38 = sub_FDD0F0((__int64)a1, **(_QWORD **)(v37 + 32));
          v39 = a1[8];
          v40 = *(_QWORD *)(v39 + 24LL * v38 + 8);
          *(_QWORD *)(v39 + 24 * v29 + 8) = v40;
          v41 = *(unsigned int *)(v40 + 104);
          if ( v41 + 1 > (unsigned __int64)*(unsigned int *)(v40 + 108) )
          {
            sub_C8D5F0(v40 + 96, (const void *)(v40 + 112), v41 + 1, 4u, v4, v5);
            v41 = *(unsigned int *)(v40 + 104);
          }
          *(_DWORD *)(*(_QWORD *)(v40 + 96) + 4 * v41) = v29;
          ++*(_DWORD *)(v40 + 104);
        }
      }
      else
      {
        v55 = 1;
        while ( v5 != -4096 )
        {
          v56 = v55 + 1;
          v35 = v34 & (v55 + v35);
          v36 = (__int64 *)(v32 + 16LL * v35);
          v5 = *v36;
          if ( v31 == *v36 )
            goto LABEL_36;
          v55 = v56;
        }
      }
    }
    goto LABEL_40;
  }
LABEL_49:
  v50 = v58;
  if ( v58 )
  {
    v51 = v63;
    v52 = v67 + 1;
    if ( v67 + 1 > v63 )
    {
      do
      {
        v53 = *v51++;
        j_j___libc_free_0(v53, 512);
      }
      while ( v52 > v51 );
      v50 = v58;
    }
    return j_j___libc_free_0(v50, 8 * v59);
  }
  return result;
}
