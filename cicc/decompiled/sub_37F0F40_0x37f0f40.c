// Function: sub_37F0F40
// Address: 0x37f0f40
//
_QWORD *__fastcall sub_37F0F40(_QWORD *a1, _QWORD *a2, __int64 a3, __int64 a4)
{
  char *v6; // r10
  char *v7; // rsi
  unsigned __int64 v9; // r13
  char **v10; // r9
  unsigned __int64 *v11; // rax
  unsigned __int64 *v12; // r9
  __int64 v13; // r14
  __int64 v14; // r8
  __int64 v15; // r12
  __int64 v16; // r11
  __int64 v17; // rcx
  __int64 v18; // rdx
  __int64 v19; // rax
  char *v20; // rdi
  __int64 v21; // rsi
  __int64 v22; // r9
  __int64 v23; // rcx
  char **v24; // rdx
  __int64 v25; // r12
  __int64 v26; // rbx
  __int64 v27; // rdx
  __int64 v28; // r14
  unsigned __int64 *v29; // r12
  __int64 v30; // rax
  __int64 v31; // rdx
  char *v32; // rsi
  __int64 v33; // rcx
  __int64 v34; // rdx
  char **v35; // rax
  char **v36; // r12
  char **v37; // rbx
  __int64 v38; // rcx
  char **v39; // r13
  unsigned __int64 *v40; // r14
  unsigned __int64 v41; // rdi
  unsigned __int64 v42; // r12
  unsigned __int64 *v43; // rbx
  unsigned __int64 *v44; // r14
  unsigned __int64 v45; // rdi
  __int64 v46; // rax
  __int64 v48; // rax
  __int64 v49; // rcx
  unsigned __int64 *v50; // rbx
  unsigned __int64 v51; // rdi
  __int64 v52; // rdx
  unsigned __int64 *v53; // rcx
  unsigned __int64 v54; // rcx
  __int64 v55; // [rsp+0h] [rbp-B0h]
  char **v56; // [rsp+8h] [rbp-A8h]
  char *v57; // [rsp+10h] [rbp-A0h]
  unsigned __int64 v58; // [rsp+18h] [rbp-98h]
  __int64 v59; // [rsp+20h] [rbp-90h]
  __int64 v60; // [rsp+20h] [rbp-90h]
  _QWORD *v61; // [rsp+20h] [rbp-90h]
  __int64 v62; // [rsp+28h] [rbp-88h]
  __int64 v63; // [rsp+28h] [rbp-88h]
  char **v64; // [rsp+30h] [rbp-80h]
  __int64 v65; // [rsp+30h] [rbp-80h]
  __int64 v66; // [rsp+30h] [rbp-80h]
  _QWORD *v67; // [rsp+30h] [rbp-80h]
  _QWORD *v68; // [rsp+30h] [rbp-80h]
  _QWORD *v69; // [rsp+30h] [rbp-80h]
  char *v70; // [rsp+38h] [rbp-78h]
  __int64 v71; // [rsp+38h] [rbp-78h]
  __int64 v72; // [rsp+38h] [rbp-78h]
  char *v73; // [rsp+40h] [rbp-70h] BYREF
  __int64 v74; // [rsp+48h] [rbp-68h]
  __int64 v75; // [rsp+50h] [rbp-60h]
  char **v76; // [rsp+58h] [rbp-58h]
  char *v77; // [rsp+60h] [rbp-50h] BYREF
  __int64 v78; // [rsp+68h] [rbp-48h]
  __int64 v79; // [rsp+70h] [rbp-40h]
  char **v80; // [rsp+78h] [rbp-38h]

  v6 = *(char **)a3;
  v7 = *(char **)a4;
  if ( *(_QWORD *)a3 == *(_QWORD *)a4 )
  {
    v46 = *(_QWORD *)(a3 + 8);
    *a1 = v6;
    a1[1] = v46;
    a1[2] = *(_QWORD *)(a3 + 16);
    a1[3] = *(_QWORD *)(a3 + 24);
    return a1;
  }
  v9 = a2[3];
  v59 = a2[6];
  v10 = (char **)a2[9];
  v70 = (char *)a2[4];
  v11 = (unsigned __int64 *)a2[5];
  v62 = a2[2];
  v64 = v10;
  if ( v6 != (char *)v62 || v7 != (char *)v59 )
  {
    v12 = *(unsigned __int64 **)(a3 + 24);
    v13 = *(_QWORD *)(a4 + 8);
    v56 = *(char ***)(a4 + 24);
    v55 = *(_QWORD *)(a3 + 16);
    v14 = *(_QWORD *)(a3 + 8);
    v15 = (((((char *)v56 - (char *)v12) >> 3) - 1) << 6) + ((__int64)&v7[-v13] >> 3) + ((v55 - (__int64)v6) >> 3);
    v16 = (__int64)&v70[-v62] >> 3;
    v58 = v16 + ((__int64)&v6[-v14] >> 3) + ((v12 - v11 - 1) << 6);
    v57 = (char *)a2[7];
    v17 = (v59 - (__int64)v57) >> 3;
    if ( (unsigned __int64)(v16 + v17 + (((((char *)v64 - (char *)v11) >> 3) - 1) << 6) - v15) >> 1 >= v58 )
    {
      if ( v6 != (char *)v62 )
      {
        v60 = *(_QWORD *)(a3 + 24);
        v18 = *(_QWORD *)(a4 + 16);
        v65 = a2[5];
        if ( v11 == v12 )
        {
          v77 = v7;
          v79 = v18;
          v78 = v13;
          v80 = v56;
          sub_37F0CB0((__int64 *)&v73, v62, v6, (__int64 *)&v77);
        }
        else
        {
          v74 = *(_QWORD *)(a4 + 8);
          v73 = v7;
          v75 = v18;
          v76 = v56;
          sub_37F0CB0((__int64 *)&v77, v14, v6, (__int64 *)&v73);
          v19 = v65;
          v20 = v77;
          v21 = v78;
          v22 = v60 - 8;
          v23 = v79;
          v24 = v80;
          if ( v65 != v60 - 8 )
          {
            v66 = v15;
            v25 = v19;
            v61 = a2;
            v26 = v22;
            do
            {
              v73 = v20;
              v26 -= 8;
              v75 = v23;
              v76 = v24;
              v74 = v21;
              sub_37F0CB0(
                (__int64 *)&v77,
                *(_QWORD *)(v26 + 8),
                (char *)(*(_QWORD *)(v26 + 8) + 512LL),
                (__int64 *)&v73);
              v20 = v77;
              v21 = v78;
              v23 = v79;
              v24 = v80;
            }
            while ( v25 != v26 );
            v15 = v66;
            a2 = v61;
          }
          v78 = v21;
          v80 = v24;
          v77 = v20;
          v79 = v23;
          sub_37F0CB0((__int64 *)&v73, v62, v70, (__int64 *)&v77);
        }
        v9 = a2[3];
        v62 = a2[2];
        v70 = (char *)a2[4];
        v11 = (unsigned __int64 *)a2[5];
      }
      v27 = v15 + ((__int64)(v62 - v9) >> 3);
      if ( v27 < 0 )
      {
        v49 = ~((unsigned __int64)~v27 >> 6);
      }
      else
      {
        if ( v27 <= 63 )
        {
          v28 = v62 + 8 * v15;
          v29 = v11;
LABEL_15:
          a2[2] = v28;
          a2[3] = v9;
          a2[4] = v70;
          a2[5] = v29;
          goto LABEL_16;
        }
        v49 = v27 >> 6;
      }
      v29 = &v11[v49];
      v9 = *v29;
      v28 = *v29 + 8 * (v27 - (v49 << 6));
      v70 = (char *)(*v29 + 512);
      if ( v29 > v11 )
      {
        v69 = a2;
        v50 = v11;
        do
        {
          v51 = *v50++;
          j_j___libc_free_0(v51);
        }
        while ( v50 < v29 );
        a2 = v69;
      }
      goto LABEL_15;
    }
    if ( v7 != (char *)v59 )
    {
      v31 = *(_QWORD *)(a4 + 16);
      if ( v56 == v64 )
      {
        v77 = v6;
        v78 = v14;
        v79 = v55;
        v80 = (char **)v12;
        sub_37F0E10(&v73, v7, v59, &v77);
      }
      else
      {
        v74 = v14;
        v73 = v6;
        v75 = v55;
        v76 = (char **)v12;
        sub_37F0E10(&v77, v7, v31, &v73);
        v32 = v77;
        v33 = v78;
        v34 = v79;
        v35 = v80;
        if ( v64 != v56 + 1 )
        {
          v71 = v15;
          v36 = v64;
          v67 = a2;
          v37 = v56 + 1;
          do
          {
            v74 = v33;
            ++v37;
            v75 = v34;
            v76 = v35;
            v73 = v32;
            sub_37F0E10(&v77, *(v37 - 1), (__int64)(*(v37 - 1) + 512), &v73);
            v32 = v77;
            v33 = v78;
            v34 = v79;
            v35 = v80;
          }
          while ( v36 != v37 );
          v15 = v71;
          a2 = v67;
        }
        v77 = v32;
        v79 = v34;
        v78 = v33;
        v80 = v35;
        sub_37F0E10(&v73, v57, v59, &v77);
      }
      v59 = a2[6];
      v57 = (char *)a2[7];
      v64 = (char **)a2[9];
      v17 = (v59 - (__int64)v57) >> 3;
    }
    v72 = a2[8];
    v38 = v17 - v15;
    if ( v38 < 0 )
    {
      v48 = ~((unsigned __int64)~v38 >> 6);
    }
    else
    {
      if ( v38 <= 63 )
      {
        v39 = v64;
        v63 = v59 - 8 * v15;
LABEL_30:
        v40 = (unsigned __int64 *)(v39 + 1);
        if ( v64 + 1 > v39 + 1 )
        {
          do
          {
            v41 = *v40++;
            j_j___libc_free_0(v41);
          }
          while ( v64 + 1 > (char **)v40 );
        }
        v28 = a2[2];
        a2[9] = v39;
        v29 = (unsigned __int64 *)a2[5];
        v9 = a2[3];
        a2[6] = v63;
        a2[7] = v57;
        a2[8] = v72;
        v70 = (char *)a2[4];
LABEL_16:
        *a1 = v28;
        a1[1] = v9;
        a1[2] = v70;
        a1[3] = v29;
        v30 = v58 + ((__int64)(v28 - v9) >> 3);
        if ( v30 < 0 )
        {
          v52 = ~((unsigned __int64)~v30 >> 6);
        }
        else
        {
          if ( v30 <= 63 )
          {
            *a1 = v28 + 8 * v58;
            return a1;
          }
          v52 = v30 >> 6;
        }
        v53 = &v29[v52];
        a1[3] = v53;
        v54 = *v53;
        a1[1] = v54;
        a1[2] = v54 + 512;
        *a1 = v54 + 8 * (v30 - (v52 << 6));
        return a1;
      }
      v48 = v38 >> 6;
    }
    v39 = &v64[v48];
    v57 = *v39;
    v72 = (__int64)(*v39 + 512);
    v63 = (__int64)&(*v39)[8 * (v38 - (v48 << 6))];
    goto LABEL_30;
  }
  v42 = (unsigned __int64)(v10 + 1);
  if ( v10 + 1 > (char **)v11 + 1 )
  {
    v68 = a2;
    v43 = v11 + 1;
    v44 = v11;
    do
    {
      v45 = *v43++;
      j_j___libc_free_0(v45);
    }
    while ( v42 > (unsigned __int64)v43 );
    a2 = v68;
    v11 = v44;
  }
  a2[7] = v9;
  a2[9] = v11;
  a2[6] = v62;
  a2[8] = v70;
  *a1 = v62;
  a1[1] = v9;
  a1[2] = v70;
  a1[3] = v11;
  return a1;
}
