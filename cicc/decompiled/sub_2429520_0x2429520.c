// Function: sub_2429520
// Address: 0x2429520
//
void __fastcall sub_2429520(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        unsigned __int64 *a6,
        __int64 a7)
{
  unsigned __int64 *v7; // r12
  unsigned __int64 *v8; // rbx
  __int64 v9; // rax
  __int64 v10; // r14
  __int64 v11; // r15
  __int64 v12; // r13
  __int64 v13; // rax
  __int64 v14; // r11
  __int64 v15; // r10
  __int64 v16; // r8
  __int64 v17; // r14
  unsigned __int64 *v18; // r12
  unsigned __int64 *v19; // rbx
  __int64 v20; // r13
  unsigned __int64 *v21; // r15
  unsigned __int64 v22; // rcx
  unsigned __int64 v23; // rdi
  unsigned __int64 *v24; // rbx
  __int64 v25; // rax
  __int64 v26; // rdx
  unsigned __int64 *v27; // rsi
  unsigned __int64 *v28; // r15
  unsigned __int64 *v29; // r14
  unsigned __int64 *i; // r13
  unsigned __int64 v31; // rdi
  unsigned __int64 v32; // rdx
  unsigned __int64 v33; // rax
  unsigned int v34; // esi
  bool v35; // cf
  unsigned __int64 v36; // rdi
  __int64 v37; // r15
  unsigned __int64 *v38; // r14
  __int64 v39; // r12
  unsigned __int64 v40; // rax
  unsigned __int64 v41; // rdi
  unsigned __int64 *v42; // rbx
  __int64 v43; // r13
  unsigned __int64 *v44; // r15
  unsigned __int64 *v45; // r14
  __int64 v46; // rbx
  unsigned __int64 v47; // rax
  unsigned __int64 v48; // rdi
  __int64 v49; // rcx
  unsigned __int64 *v50; // r15
  unsigned __int64 v51; // rax
  unsigned __int64 v52; // rdx
  unsigned int v53; // ecx
  unsigned __int64 v54; // rdi
  unsigned __int64 v55; // rdi
  __int64 v56; // rax
  __int64 v57; // rax
  __int64 v58; // rbx
  unsigned __int64 *v59; // r15
  unsigned __int64 *v60; // r13
  unsigned __int64 v61; // rax
  unsigned __int64 v62; // rdi
  __int64 v63; // rbx
  unsigned __int64 *v64; // r12
  __int64 v65; // r13
  unsigned __int64 v66; // rax
  unsigned __int64 v67; // rdi
  __int64 v68; // [rsp+10h] [rbp-60h]
  __int64 v69; // [rsp+18h] [rbp-58h]
  __int64 v70; // [rsp+20h] [rbp-50h]
  __int64 v71; // [rsp+20h] [rbp-50h]
  __int64 v72; // [rsp+28h] [rbp-48h]
  __int64 v73; // [rsp+28h] [rbp-48h]
  unsigned __int64 *v74; // [rsp+28h] [rbp-48h]
  __int64 v75; // [rsp+30h] [rbp-40h]
  __int64 v76; // [rsp+30h] [rbp-40h]
  __int64 v77; // [rsp+30h] [rbp-40h]
  __int64 v78; // [rsp+38h] [rbp-38h]

  while ( 1 )
  {
    v7 = (unsigned __int64 *)a1;
    v8 = a6;
    v78 = a3;
    v9 = a7;
    if ( a5 <= a7 )
      v9 = a5;
    if ( v9 >= a4 )
    {
      v43 = a2;
      v44 = (unsigned __int64 *)a1;
      v45 = a6;
      v77 = a2 - a1;
      if ( a2 - a1 <= 0 )
        return;
      v74 = a6;
      v46 = (a2 - a1) >> 3;
      do
      {
        v47 = *v44;
        *v44 = 0;
        v48 = *v45;
        *v45 = v47;
        if ( v48 )
          j_j___libc_free_0(v48);
        ++v44;
        ++v45;
        --v46;
      }
      while ( v46 );
      v49 = v77;
      v42 = v74;
      if ( v77 <= 0 )
        v49 = 8;
      v50 = (unsigned __int64 *)((char *)v74 + v49);
      if ( v74 == (unsigned __int64 *)((char *)v74 + v49) )
        return;
      while ( 1 )
      {
        if ( v78 == v43 )
        {
          v37 = (char *)v50 - (char *)v42;
          v38 = v7;
          v39 = v37 >> 3;
          if ( v37 > 0 )
          {
            do
            {
              v40 = *v42;
              *v42 = 0;
              v41 = *v38;
              *v38 = v40;
              if ( v41 )
                j_j___libc_free_0(v41);
              ++v42;
              ++v38;
              --v39;
            }
            while ( v39 );
          }
          return;
        }
        v51 = *v42;
        v52 = *(_QWORD *)v43;
        v53 = *(_DWORD *)(*v42 + 32);
        if ( *(_DWORD *)(*(_QWORD *)v43 + 32LL) == v53 )
        {
          if ( *(_DWORD *)(v52 + 36) >= *(_DWORD *)(v51 + 36) )
          {
LABEL_53:
            *v42 = 0;
            v55 = *v7;
            *v7 = v51;
            if ( v55 )
              j_j___libc_free_0(v55);
            ++v42;
            goto LABEL_50;
          }
        }
        else if ( *(_DWORD *)(*(_QWORD *)v43 + 32LL) >= v53 )
        {
          goto LABEL_53;
        }
        *(_QWORD *)v43 = 0;
        v54 = *v7;
        *v7 = v52;
        if ( v54 )
          j_j___libc_free_0(v54);
        v43 += 8;
LABEL_50:
        ++v7;
        if ( v42 == v50 )
          return;
      }
    }
    v10 = a2;
    v11 = a5;
    if ( a5 <= a7 )
      break;
    if ( a5 >= a4 )
    {
      v71 = a5 / 2;
      v72 = a2 + 8 * (a5 / 2);
      v56 = sub_2425A60(a1, a2, v72);
      v16 = v71;
      v12 = v56;
      v75 = (v56 - a1) >> 3;
    }
    else
    {
      v75 = a4 / 2;
      v12 = a1 + 8 * (a4 / 2);
      v13 = sub_2425A00(a2, a3, v12);
      v15 = a7;
      v72 = v13;
      v16 = (v13 - a2) >> 3;
    }
    v68 = v14 - v75;
    v69 = v15;
    v70 = v16;
    v17 = sub_2428400(v12, a2, v72, v14 - v75, v16, v8, v15);
    sub_2429520(a1, v12, v17, v75, v70, (_DWORD)v8, v69);
    a6 = v8;
    a2 = v72;
    a1 = v17;
    a7 = v69;
    a3 = v78;
    a5 = v11 - v70;
    a4 = v68;
  }
  if ( a3 - a2 <= 0 )
    return;
  v73 = a1;
  v18 = a6;
  v19 = (unsigned __int64 *)a2;
  v76 = a3 - a2;
  v20 = (a3 - a2) >> 3;
  v21 = a6;
  do
  {
    v22 = *v19;
    *v19 = 0;
    v23 = *v18;
    *v18 = v22;
    if ( v23 )
      j_j___libc_free_0(v23);
    ++v19;
    ++v18;
    --v20;
  }
  while ( v20 );
  v24 = v21;
  v25 = 8;
  v26 = v76 - 8;
  if ( v76 <= 0 )
    v26 = 0;
  else
    v25 = v76;
  v27 = (unsigned __int64 *)((char *)v21 + v25);
  v28 = (unsigned __int64 *)((char *)v21 + v26);
  if ( v73 != v10 )
  {
    if ( v24 == v27 )
      return;
    v29 = (unsigned __int64 *)(v10 - 8);
    for ( i = (unsigned __int64 *)(v78 - 8); ; --i )
    {
      v32 = *v29;
      v33 = *v28;
      v34 = *(_DWORD *)(*v29 + 32);
      v35 = *(_DWORD *)(*v28 + 32) < v34;
      if ( *(_DWORD *)(*v28 + 32) == v34 )
        v35 = *(_DWORD *)(v33 + 36) < *(_DWORD *)(v32 + 36);
      if ( v35 )
      {
        *v29 = 0;
        v31 = *i;
        *i = v32;
        if ( v31 )
          j_j___libc_free_0(v31);
        if ( (unsigned __int64 *)v73 == v29 )
        {
          v57 = (char *)(v28 + 1) - (char *)v24;
          v58 = v57 >> 3;
          if ( v57 > 0 )
          {
            v59 = &v28[-v58];
            v60 = &i[-v58];
            do
            {
              v61 = v59[v58];
              v59[v58] = 0;
              v62 = v60[v58 - 1];
              v60[v58 - 1] = v61;
              if ( v62 )
                j_j___libc_free_0(v62);
              --v58;
            }
            while ( v58 );
          }
          return;
        }
        --v29;
      }
      else
      {
        *v28 = 0;
        v36 = *i;
        *i = v33;
        if ( v36 )
          j_j___libc_free_0(v36);
        if ( v24 == v28 )
          return;
        --v28;
      }
    }
  }
  v63 = v25 >> 3;
  v64 = &v27[-(v25 >> 3)];
  v65 = -8 * (v25 >> 3) + v78;
  do
  {
    v66 = v64[v63 - 1];
    v64[v63 - 1] = 0;
    v67 = *(_QWORD *)(v65 + 8 * v63 - 8);
    *(_QWORD *)(v65 + 8 * v63 - 8) = v66;
    if ( v67 )
      j_j___libc_free_0(v67);
    --v63;
  }
  while ( v63 );
}
