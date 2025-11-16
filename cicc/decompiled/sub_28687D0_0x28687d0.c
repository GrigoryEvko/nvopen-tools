// Function: sub_28687D0
// Address: 0x28687d0
//
_BOOL8 __fastcall sub_28687D0(__int64 *a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rcx
  __int64 v5; // r13
  __int64 v6; // rbx
  _QWORD *v7; // rax
  __int64 v8; // r8
  __int64 v9; // r9
  _QWORD *v10; // r13
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rdx
  __int64 v16; // rdx
  int v17; // ebx
  unsigned __int64 v18; // rax
  unsigned __int64 v19; // rsi
  unsigned __int64 v20; // rcx
  __int64 v21; // r12
  _QWORD *v22; // rax
  _QWORD *i; // r12
  __int64 v24; // r8
  __int64 *v25; // r9
  __int64 v26; // rbx
  unsigned __int8 *v27; // r15
  unsigned __int64 v28; // r13
  __int64 v29; // r12
  _QWORD *v30; // rbx
  _QWORD *v31; // rax
  __int64 v32; // r12
  __int64 *v33; // rax
  __int64 v34; // r8
  __int64 v35; // r9
  __int64 v36; // rdx
  unsigned __int64 v37; // r13
  __int64 v38; // rcx
  unsigned __int64 v39; // rsi
  unsigned __int64 v40; // rdx
  unsigned __int64 *v41; // rdi
  __int64 v42; // rdx
  __int64 v43; // rbx
  __int64 v44; // rdx
  unsigned __int64 v45; // rcx
  unsigned __int64 v46; // rsi
  int v47; // eax
  __int64 v48; // rsi
  unsigned __int64 *v49; // rcx
  unsigned __int64 *v50; // rdx
  _QWORD *v52; // rax
  __int64 v53; // r12
  _QWORD *v54; // rbx
  _QWORD *v55; // rax
  int v56; // r12d
  __int64 v57; // r15
  __int64 *v58; // rax
  __int64 v59; // rcx
  __int64 v60; // r12
  __int64 v61; // r15
  unsigned __int64 *v62; // r8
  unsigned __int64 v63; // rdi
  __int64 v64; // rdi
  char *v65; // r12
  __int64 v66; // r13
  __int64 v67; // rdi
  int v68; // edx
  bool v69; // [rsp+10h] [rbp-90h]
  __int64 *v70; // [rsp+10h] [rbp-90h]
  _QWORD *v72; // [rsp+28h] [rbp-78h]
  __int64 v73; // [rsp+28h] [rbp-78h]
  unsigned __int64 *v74; // [rsp+28h] [rbp-78h]
  unsigned __int64 *v75; // [rsp+28h] [rbp-78h]
  __int64 v76; // [rsp+28h] [rbp-78h]
  _QWORD *v77; // [rsp+38h] [rbp-68h] BYREF
  unsigned __int8 *v78; // [rsp+40h] [rbp-60h] BYREF
  unsigned __int8 *v79; // [rsp+48h] [rbp-58h]
  __int64 v80; // [rsp+50h] [rbp-50h] BYREF
  _QWORD *v81; // [rsp+58h] [rbp-48h]
  __int64 v82; // [rsp+60h] [rbp-40h]

  v3 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  v4 = *(_QWORD *)(*(_QWORD *)(a2 - 32 * v3) + 24LL);
  v78 = (unsigned __int8 *)v4;
  if ( *(_BYTE *)v4 == 4 )
  {
    if ( !*(_DWORD *)(v4 + 144) && !(unsigned __int8)sub_AF4500(*(_QWORD *)(*(_QWORD *)(a2 + 32 * (2 - v3)) + 24LL)) )
      return 0;
  }
  else if ( (unsigned __int8)(*(_BYTE *)v4 - 5) <= 0x1Fu )
  {
    return 0;
  }
  sub_B58DC0(&v80, &v78);
  v69 = sub_2850EF0(&v80);
  if ( v69 )
    return 0;
  v5 = *a1;
  sub_B58E30(&v80, a2);
  v6 = v80;
  v72 = v81;
  if ( v81 == (_QWORD *)v80 )
  {
LABEL_5:
    v7 = (_QWORD *)sub_22077B0(0x98u);
    v10 = v7;
    if ( v7 )
    {
      v77 = v7;
      *v7 = a2 & 0xFFFFFFFFFFFFFFFBLL;
      v11 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
      v12 = 2 - v11;
      v13 = *(_QWORD *)(a2 - 32 * v11);
      v14 = *(_QWORD *)(*(_QWORD *)(a2 + 32 * v12) + 24LL);
      v10[11] = v10 + 13;
      *((_BYTE *)v10 + 16) = 0;
      v10[1] = v14;
      v10[3] = v10 + 5;
      v10[4] = 0x200000000LL;
      v10[12] = 0x200000000LL;
      v10[15] = v10 + 17;
      v10[16] = 0x200000000LL;
      v15 = *(_QWORD *)(v13 + 24);
      if ( *(_BYTE *)v15 != 4 )
      {
        v16 = 8;
        v17 = 1;
        v18 = 0;
        v19 = 1;
        goto LABEL_8;
      }
      v18 = 0;
    }
    else
    {
      v68 = *(_DWORD *)(a2 + 4);
      v18 = MEMORY[0x80];
      v77 = 0;
      v15 = *(_QWORD *)(*(_QWORD *)(a2 - 32LL * (v68 & 0x7FFFFFF)) + 24LL);
      if ( *(_BYTE *)v15 != 4 )
      {
        v19 = 1;
        v17 = 1;
LABEL_56:
        if ( v18 == v19 )
          goto LABEL_15;
        v16 = 8 * v19;
        if ( v18 > v19 )
        {
          v59 = v10[15];
          v60 = v59 + 8 * v18;
          v61 = v59 + v16;
          while ( v61 != v60 )
          {
            v62 = *(unsigned __int64 **)(v60 - 8);
            v60 -= 8;
            if ( v62 )
            {
              v63 = v62[8];
              if ( (unsigned __int64 *)v63 != v62 + 10 )
              {
                v74 = v62;
                _libc_free(v63);
                v62 = v74;
              }
              if ( (unsigned __int64 *)*v62 != v62 + 2 )
              {
                v75 = v62;
                _libc_free(*v62);
                v62 = v75;
              }
              j_j___libc_free_0((unsigned __int64)v62);
            }
          }
          goto LABEL_14;
        }
LABEL_8:
        v20 = *((unsigned int *)v10 + 33);
        if ( v19 > v20 )
        {
          v76 = v16;
          sub_2851110((__int64)(v10 + 15), v19, v16, v20, v8, v9);
          v18 = *((unsigned int *)v10 + 32);
          v16 = v76;
        }
        v21 = v10[15];
        v22 = (_QWORD *)(v21 + 8 * v18);
        for ( i = (_QWORD *)(v16 + v21); i != v22; ++v22 )
        {
          if ( v22 )
            *v22 = 0;
        }
LABEL_14:
        *((_DWORD *)v10 + 32) = v17;
LABEL_15:
        sub_B58E30(&v78, a2);
        v26 = (__int64)v78;
        v27 = v79;
        if ( v79 != v78 )
        {
          v28 = (unsigned __int64)v77;
          do
          {
            while ( 1 )
            {
              v29 = v26;
              v30 = (_QWORD *)(v26 & 0xFFFFFFFFFFFFFFF8LL);
              v31 = v30;
              LODWORD(v29) = (v29 >> 2) & 1;
              v73 = (unsigned int)v29;
              if ( (_DWORD)v29 )
                v31 = (_QWORD *)*v30;
              v32 = v31[17];
              v33 = sub_DD8400(*a1, v32);
              v36 = *(unsigned int *)(v28 + 96);
              if ( v36 + 1 > (unsigned __int64)*(unsigned int *)(v28 + 100) )
              {
                v70 = v33;
                sub_C8D5F0(v28 + 88, (const void *)(v28 + 104), v36 + 1, 8u, v34, v35);
                v36 = *(unsigned int *)(v28 + 96);
                v33 = v70;
              }
              *(_QWORD *)(*(_QWORD *)(v28 + 88) + 8 * v36) = v33;
              ++*(_DWORD *)(v28 + 96);
              v37 = (unsigned __int64)v77;
              v80 = 4;
              v81 = 0;
              v82 = v32;
              if ( v32 != 0 && v32 != -4096 && v32 != -8192 )
                sub_BD73F0((__int64)&v80);
              v38 = *((unsigned int *)v77 + 8);
              v39 = v77[3];
              v25 = &v80;
              v24 = v38 + 1;
              v40 = v38;
              if ( v38 + 1 > (unsigned __int64)*((unsigned int *)v77 + 9) )
              {
                v64 = (__int64)(v77 + 3);
                if ( v39 > (unsigned __int64)&v80 || (v40 = v39 + 24 * v38, (unsigned __int64)&v80 >= v40) )
                {
                  sub_D6B130(v64, v38 + 1, v40, v38, v24, (__int64)&v80);
                  v38 = *((unsigned int *)v77 + 8);
                  v39 = v77[3];
                  v25 = &v80;
                  LODWORD(v40) = *((_DWORD *)v77 + 8);
                }
                else
                {
                  v65 = (char *)&v80 - v39;
                  sub_D6B130(v64, v38 + 1, v40, v38, v24, (__int64)&v80);
                  v39 = v77[3];
                  v38 = *((unsigned int *)v77 + 8);
                  v25 = (__int64 *)&v65[v39];
                  LODWORD(v40) = *((_DWORD *)v77 + 8);
                }
              }
              v41 = (unsigned __int64 *)(v39 + 24 * v38);
              if ( v41 )
              {
                *v41 = 4;
                v42 = v25[2];
                v41[1] = 0;
                v41[2] = v42;
                if ( v42 != -4096 && v42 != 0 && v42 != -8192 )
                  sub_BD6050(v41, *v25 & 0xFFFFFFFFFFFFFFF8LL);
                LODWORD(v40) = *(_DWORD *)(v37 + 32);
              }
              *(_DWORD *)(v37 + 32) = v40 + 1;
              if ( v82 != -4096 && v82 != 0 && v82 != -8192 )
                sub_BD60C0(&v80);
              v28 = (unsigned __int64)v77;
              *((_BYTE *)v77 + 16) = **(_BYTE **)(*(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)) + 24LL) == 4;
              if ( v73 )
                break;
              v26 = (__int64)(v30 + 18);
              if ( v27 == (unsigned __int8 *)v26 )
                goto LABEL_36;
            }
            v26 = (unsigned __int64)(v30 + 1) | 4;
          }
          while ( v27 != (unsigned __int8 *)v26 );
        }
LABEL_36:
        v43 = a1[1];
        v44 = *(unsigned int *)(v43 + 8);
        v45 = *(unsigned int *)(v43 + 12);
        v46 = v44 + 1;
        v47 = *(_DWORD *)(v43 + 8);
        if ( v44 + 1 > v45 )
        {
          v66 = *(_QWORD *)v43;
          v67 = a1[1];
          if ( *(_QWORD *)v43 > (unsigned __int64)&v77 || (unsigned __int64)&v77 >= v66 + 8 * v44 )
          {
            sub_2853040(v67, v46, v44, v45, v24, (__int64)v25);
            v44 = *(unsigned int *)(v43 + 8);
            v48 = *(_QWORD *)v43;
            v49 = (unsigned __int64 *)&v77;
            v47 = *(_DWORD *)(v43 + 8);
          }
          else
          {
            sub_2853040(v67, v46, v44, v45, v24, (__int64)v25);
            v48 = *(_QWORD *)v43;
            v44 = *(unsigned int *)(v43 + 8);
            v49 = (unsigned __int64 *)((char *)&v77 + *(_QWORD *)v43 - v66);
            v47 = *(_DWORD *)(v43 + 8);
          }
        }
        else
        {
          v48 = *(_QWORD *)v43;
          v49 = (unsigned __int64 *)&v77;
        }
        v50 = (unsigned __int64 *)(v48 + 8 * v44);
        if ( v50 )
        {
          *v50 = *v49;
          *v49 = 0;
          v47 = *(_DWORD *)(v43 + 8);
        }
        *(_DWORD *)(v43 + 8) = v47 + 1;
        sub_28548B0((unsigned __int64 *)&v77);
        return 1;
      }
    }
    v19 = *(unsigned int *)(v15 + 144);
    v17 = *(_DWORD *)(v15 + 144);
    goto LABEL_56;
  }
  while ( 1 )
  {
    v53 = v6;
    v54 = (_QWORD *)(v6 & 0xFFFFFFFFFFFFFFF8LL);
    v55 = v54;
    v56 = (v53 >> 2) & 1;
    if ( v56 )
      v55 = (_QWORD *)*v54;
    v57 = v55[17];
    if ( !v57 )
      return v69;
    if ( !sub_D97040(v5, *(_QWORD *)(v57 + 8)) )
      return v69;
    v58 = sub_DD8400(v5, v57);
    if ( (unsigned __int8)sub_D98260(v5, (__int64)v58) )
      return v69;
    if ( v56 )
    {
      v6 = (unsigned __int64)(v54 + 1) | 4;
      v52 = (_QWORD *)v6;
    }
    else
    {
      v52 = v54 + 18;
      v6 = (__int64)(v54 + 18);
    }
    if ( v52 == v72 )
      goto LABEL_5;
  }
}
