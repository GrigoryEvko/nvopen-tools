// Function: sub_2A5BFF0
// Address: 0x2a5bff0
//
__int64 **__fastcall sub_2A5BFF0(__int64 **a1, __int64 *a2)
{
  __int64 *v3; // rbx
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rax
  unsigned __int64 v7; // rdi
  __int64 v8; // rbx
  __int64 v9; // rdx
  __int64 v10; // rax
  unsigned __int64 v11; // rsi
  _QWORD *v12; // rax
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // r8
  unsigned __int64 v16; // rsi
  _QWORD *v17; // rax
  __int64 v18; // rax
  __int64 v19; // r9
  __int64 v20; // rsi
  __int64 v21; // r8
  __int64 *v22; // r8
  __int64 *v23; // rax
  __int64 *v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // rsi
  __int64 *v27; // r15
  __int64 *v28; // r11
  __int64 v29; // rax
  __int64 v30; // r8
  _QWORD *v31; // rdx
  __int64 v32; // r14
  __int64 *v33; // rdx
  __int64 v34; // rbx
  __int64 v35; // r12
  __int64 v36; // rdx
  __int64 v37; // rax
  unsigned __int64 v38; // rdi
  unsigned __int64 *v39; // rbx
  unsigned __int64 *v40; // r12
  unsigned __int64 v41; // rdi
  __int64 v43; // rdx
  __int64 *v44; // rsi
  __int64 v45; // rcx
  _QWORD *v46; // rdx
  __int64 v47; // rdx
  const void *v48; // r10
  signed __int64 v49; // rdx
  __int64 v50; // rax
  __int64 v51; // rsi
  bool v52; // cf
  unsigned __int64 v53; // rax
  char *v54; // r9
  char *v55; // rax
  unsigned __int64 v56; // rsi
  __int64 v57; // rax
  __int64 v58; // [rsp+8h] [rbp-C8h]
  __int64 v59; // [rsp+8h] [rbp-C8h]
  __int64 v60; // [rsp+8h] [rbp-C8h]
  __int64 *v61; // [rsp+10h] [rbp-C0h]
  __int64 *v62; // [rsp+10h] [rbp-C0h]
  __int64 *v63; // [rsp+10h] [rbp-C0h]
  __int64 v64; // [rsp+18h] [rbp-B8h]
  char *v65; // [rsp+18h] [rbp-B8h]
  signed __int64 v66; // [rsp+18h] [rbp-B8h]
  const void *v67; // [rsp+20h] [rbp-B0h]
  __int64 v68; // [rsp+20h] [rbp-B0h]
  const void *v69; // [rsp+20h] [rbp-B0h]
  __int64 v70; // [rsp+28h] [rbp-A8h]
  __int64 v71; // [rsp+28h] [rbp-A8h]
  __int64 v72; // [rsp+28h] [rbp-A8h]
  unsigned __int64 v73; // [rsp+30h] [rbp-A0h]
  int v75; // [rsp+44h] [rbp-8Ch] BYREF
  __int64 v76; // [rsp+48h] [rbp-88h] BYREF
  unsigned __int64 v77[2]; // [rsp+50h] [rbp-80h] BYREF
  __int64 v78; // [rsp+60h] [rbp-70h]
  __int64 v79; // [rsp+68h] [rbp-68h]
  __int64 v80; // [rsp+70h] [rbp-60h]
  unsigned __int64 v81; // [rsp+78h] [rbp-58h]
  __int64 v82; // [rsp+80h] [rbp-50h]
  __int64 v83; // [rsp+88h] [rbp-48h]
  __int64 v84; // [rsp+90h] [rbp-40h]
  __int64 *v85; // [rsp+98h] [rbp-38h]

  v78 = 0;
  v81 = 0;
  v82 = 0;
  v85 = 0;
  v77[1] = 8;
  v77[0] = sub_22077B0(0x40u);
  v3 = (__int64 *)(v77[0] + 24);
  v4 = sub_22077B0(0x200u);
  *a1 = 0;
  a1[1] = 0;
  a1[2] = 0;
  *v3 = v4;
  v79 = v4;
  v80 = v4 + 512;
  v83 = v4;
  v84 = v4 + 512;
  v5 = a2[1];
  v78 = v4;
  v82 = v4;
  v6 = *a2;
  v81 = (unsigned __int64)v3;
  v85 = v3;
  if ( v6 != v5 )
  {
    do
    {
      *(_QWORD *)(v6 + 48) = 0;
      v6 += 72;
      *(_QWORD *)(v6 - 16) = 0;
      *(_QWORD *)(v6 - 8) = 0;
      *(_BYTE *)(v6 - 48) = 0;
    }
    while ( v6 != v5 );
    v5 = *a2;
  }
  *(_BYTE *)(v5 + 72 * a2[7] + 24) = 1;
  LODWORD(v76) = 0;
  sub_2A5B6F0(v77, a2 + 6, (int *)&v76);
  *(_QWORD *)(*a2 + 72 * a2[6] + 48) = 1;
  v7 = v82;
  if ( v82 != v78 )
  {
    v8 = 1;
    do
    {
      while ( 1 )
      {
        if ( v83 == v7 )
        {
          v9 = *(_QWORD *)(*(v85 - 1) + 496);
          v76 = v9;
          v15 = *(v85 - 1);
          v16 = *(_QWORD *)(v15 + 504);
          v17 = (_QWORD *)(a2[3] + 24 * v9);
          if ( 0x6DB6DB6DB6DB6DB7LL * ((__int64)(v17[1] - *v17) >> 3) <= v16 )
          {
            j_j___libc_free_0(v7);
            v47 = *--v85 + 512;
            v83 = *v85;
            v84 = v47;
            v82 = v83 + 496;
LABEL_8:
            v13 = v76;
            v14 = *a2 + 72 * v76;
            if ( *(_BYTE *)(v14 + 24) )
            {
              *(_QWORD *)(v14 + 56) = ++v8;
              if ( v13 != a2[6] )
              {
                v43 = v82;
                if ( v82 == v83 )
                  v43 = *(v85 - 1) + 512;
                *(_BYTE *)(*a2 + 72LL * *(_QWORD *)(v43 - 16) + 24) = 1;
              }
              v44 = a1[1];
              if ( v44 == a1[2] )
              {
                sub_9CA200((__int64)a1, v44, &v76);
              }
              else
              {
                if ( v44 )
                {
                  *v44 = v13;
                  v44 = a1[1];
                }
                a1[1] = v44 + 1;
              }
            }
            else
            {
              *(_QWORD *)(v14 + 48) = 0;
            }
            goto LABEL_10;
          }
          v7 = v15 + 512;
          v18 = *v17 + 56 * v16;
          v19 = *a2;
          v20 = *(_QWORD *)(v18 + 24);
        }
        else
        {
          v10 = a2[3];
          v76 = *(_QWORD *)(v7 - 16);
          v9 = v76;
          v11 = *(_QWORD *)(v7 - 8);
          v12 = (_QWORD *)(v10 + 24 * v76);
          if ( 0x6DB6DB6DB6DB6DB7LL * ((__int64)(v12[1] - *v12) >> 3) <= v11 )
          {
            v82 = v7 - 16;
            goto LABEL_8;
          }
          v19 = *a2;
          v18 = *v12 + 56 * v11;
          v20 = *(_QWORD *)(v18 + 24);
        }
        ++*(_QWORD *)(v7 - 8);
        if ( *(_BYTE *)(v18 + 40) )
          break;
LABEL_10:
        v7 = v82;
        if ( v78 == v82 )
          goto LABEL_20;
      }
      v21 = v19 + 72 * v20;
      if ( !*(_QWORD *)(v21 + 48) && *(_QWORD *)(v21 + 64) <= 9u )
      {
        *(_QWORD *)(v21 + 48) = ++v8;
        v71 = v19 + 72 * v20;
        v75 = 0;
        sub_2A5B6F0(v77, (_QWORD *)(v18 + 24), &v75);
        ++*(_QWORD *)(v71 + 64);
        goto LABEL_10;
      }
      if ( !*(_BYTE *)(v21 + 24) || !*(_QWORD *)(v21 + 56) )
        goto LABEL_10;
      *(_BYTE *)(*a2 + 72 * v9 + 24) = 1;
      v7 = v82;
    }
    while ( v78 != v82 );
  }
LABEL_20:
  v22 = a1[1];
  v23 = *a1;
  if ( *a1 != v22 )
  {
    v24 = v22 - 1;
    if ( v23 >= v22 - 1 )
      goto LABEL_24;
    do
    {
      v25 = *v23;
      v26 = *v24;
      ++v23;
      --v24;
      *(v23 - 1) = v26;
      v24[1] = v25;
    }
    while ( v24 > v23 );
    v22 = a1[1];
    v23 = *a1;
    if ( v22 != *a1 )
    {
LABEL_24:
      v27 = v23;
      v28 = v22;
      while ( 1 )
      {
        v29 = *v27;
        v30 = 24 * *v27;
        v31 = (_QWORD *)(v30 + a2[8]);
        if ( *v31 != v31[1] )
          v31[1] = *v31;
        v32 = 72 * v29;
        v33 = (__int64 *)(v30 + a2[3]);
        v34 = *v33;
        v35 = v33[1];
        if ( v35 != *v33 )
          break;
LABEL_33:
        if ( v28 == ++v27 )
          goto LABEL_34;
      }
      while ( 1 )
      {
        if ( !*(_BYTE *)(v34 + 40) )
          goto LABEL_32;
        v36 = *a2 + v32;
        if ( !*(_BYTE *)(v36 + 24) )
          goto LABEL_32;
        v37 = *a2 + 72LL * *(_QWORD *)(v34 + 24);
        if ( !*(_BYTE *)(v37 + 24) || *(_QWORD *)(v37 + 56) >= *(_QWORD *)(v36 + 56) )
          goto LABEL_32;
        v45 = v30 + a2[8];
        v46 = *(_QWORD **)(v45 + 8);
        if ( v46 != *(_QWORD **)(v45 + 16) )
        {
          if ( v46 )
          {
            *v46 = v34;
            v46 = *(_QWORD **)(v45 + 8);
          }
          *(_QWORD *)(v45 + 8) = v46 + 1;
          goto LABEL_32;
        }
        v48 = *(const void **)v45;
        v49 = (signed __int64)v46 - *(_QWORD *)v45;
        v50 = v49 >> 3;
        if ( v49 >> 3 == 0xFFFFFFFFFFFFFFFLL )
          sub_4262D8((__int64)"vector::_M_realloc_insert");
        v51 = 1;
        if ( v50 )
          v51 = v49 >> 3;
        v52 = __CFADD__(v51, v50);
        v53 = v51 + v50;
        if ( v52 )
        {
          v56 = 0x7FFFFFFFFFFFFFF8LL;
        }
        else
        {
          if ( !v53 )
          {
            v73 = 0;
            v54 = 0;
            goto LABEL_60;
          }
          if ( v53 > 0xFFFFFFFFFFFFFFFLL )
            v53 = 0xFFFFFFFFFFFFFFFLL;
          v56 = 8 * v53;
        }
        v60 = v30;
        v63 = v28;
        v66 = v49;
        v69 = *(const void **)v45;
        v72 = v30 + a2[8];
        v57 = sub_22077B0(v56);
        v45 = v72;
        v54 = (char *)v57;
        v48 = v69;
        v49 = v66;
        v28 = v63;
        v30 = v60;
        v73 = v56 + v57;
LABEL_60:
        if ( &v54[v49] )
          *(_QWORD *)&v54[v49] = v34;
        v70 = (__int64)&v54[v49 + 8];
        if ( v49 > 0 )
        {
          v58 = v30;
          v61 = v28;
          v64 = v45;
          v67 = v48;
          v55 = (char *)memmove(v54, v48, v49);
          v45 = v64;
          v48 = v67;
          v28 = v61;
          v30 = v58;
          v54 = v55;
LABEL_68:
          v59 = v30;
          v62 = v28;
          v65 = v54;
          v68 = v45;
          j_j___libc_free_0((unsigned __int64)v48);
          v30 = v59;
          v28 = v62;
          v54 = v65;
          v45 = v68;
          goto LABEL_64;
        }
        if ( v48 )
          goto LABEL_68;
LABEL_64:
        *(_QWORD *)v45 = v54;
        *(_QWORD *)(v45 + 8) = v70;
        *(_QWORD *)(v45 + 16) = v73;
LABEL_32:
        v34 += 56;
        if ( v35 == v34 )
          goto LABEL_33;
      }
    }
  }
LABEL_34:
  v38 = v77[0];
  if ( v77[0] )
  {
    v39 = (unsigned __int64 *)v81;
    v40 = (unsigned __int64 *)(v85 + 1);
    if ( (unsigned __int64)(v85 + 1) > v81 )
    {
      do
      {
        v41 = *v39++;
        j_j___libc_free_0(v41);
      }
      while ( v40 > v39 );
      v38 = v77[0];
    }
    j_j___libc_free_0(v38);
  }
  return a1;
}
