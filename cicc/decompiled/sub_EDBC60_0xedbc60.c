// Function: sub_EDBC60
// Address: 0xedbc60
//
__int64 *__fastcall sub_EDBC60(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rsi
  unsigned __int64 v5; // rbx
  __int64 v6; // rsi
  __int64 v8; // rax
  __int64 v9; // rdi
  __int64 v10; // rsi
  size_t v11; // rdx
  const void *v12; // r14
  signed __int64 v13; // r13
  char *v14; // r8
  char *v15; // r13
  __int64 **v16; // r13
  _QWORD *v17; // rbx
  __int64 *v18; // rcx
  __int64 *v19; // r12
  __int64 *v20; // r13
  __int64 v21; // rax
  char *v22; // r15
  char **v23; // rbx
  __int64 *v24; // r14
  unsigned __int64 v25; // r12
  char **v26; // rsi
  char *v27; // r13
  _QWORD *v28; // rdi
  void (*v29)(void); // rax
  __int64 v30; // rdx
  unsigned __int16 *v31; // rax
  __int64 v32; // rcx
  __int64 v33; // rax
  unsigned __int64 v34; // r14
  char **v35; // rsi
  __int64 *v36; // r14
  __int64 *v37; // r12
  char *v38; // rcx
  __int64 v39; // r13
  unsigned __int64 v40; // r13
  __int64 v41; // rax
  _QWORD *v42; // r13
  _QWORD *v43; // r14
  __int64 *v44; // r15
  char *v45; // rcx
  __int64 v46; // rbx
  size_t v47; // rbx
  __int64 *v48; // r14
  __int64 *v49; // r12
  _QWORD *v50; // rbx
  unsigned __int64 v51; // r14
  _QWORD *v52; // r12
  __int64 *v53; // r14
  __int64 *v54; // r15
  char *v55; // rcx
  __int64 v56; // rbx
  size_t v57; // rbx
  _QWORD *v58; // r14
  _QWORD *v59; // r13
  _QWORD *v60; // rdi
  _QWORD *v61; // r12
  _QWORD *v62; // rbx
  char *v63; // rbx
  _QWORD *v64; // r14
  _QWORD *v65; // rdi
  _QWORD *v66; // r13
  _QWORD *v67; // r12
  _QWORD *v68; // [rsp+0h] [rbp-A0h]
  __int64 *v69; // [rsp+0h] [rbp-A0h]
  _QWORD *v70; // [rsp+0h] [rbp-A0h]
  __int64 **v71; // [rsp+8h] [rbp-98h]
  __int64 v72; // [rsp+8h] [rbp-98h]
  __int64 v73; // [rsp+18h] [rbp-88h]
  size_t v74; // [rsp+20h] [rbp-80h]
  __int64 **v76; // [rsp+30h] [rbp-70h]
  _QWORD *v77; // [rsp+30h] [rbp-70h]
  unsigned __int64 v80; // [rsp+58h] [rbp-48h] BYREF
  __int64 v81; // [rsp+60h] [rbp-40h] BYREF
  unsigned __int64 v82; // [rsp+68h] [rbp-38h]

  v4 = *(_QWORD *)(a2 + 128);
  v81 = 0;
  v82 = 0;
  (*(void (__fastcall **)(unsigned __int64 *, __int64, __int64 *))(*(_QWORD *)v4 + 16LL))(&v80, v4, &v81);
  v5 = v80 & 0xFFFFFFFFFFFFFFFELL;
  if ( (v80 & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    v6 = a2;
    v80 = v5 | 1;
    sub_EDBB60(a1, a2, (__int64 *)&v80);
    goto LABEL_3;
  }
  v80 = 0;
  v8 = *(unsigned int *)(a2 + 488);
  v9 = a3;
  *(_DWORD *)(a2 + 488) = v8 + 1;
  v10 = v81 + 80 * v8;
  v73 = v10;
  sub_ED69A0(a3, (char **)v10);
  v11 = v10 + 24;
  if ( v10 + 24 != a3 + 24 )
  {
    v11 = *(_QWORD *)(v10 + 32);
    v12 = *(const void **)(v10 + 24);
    v9 = *(_QWORD *)(a3 + 24);
    v13 = v11 - (_QWORD)v12;
    v10 = *(_QWORD *)(a3 + 40) - v9;
    if ( v11 - (unsigned __int64)v12 > v10 )
    {
      if ( v13 )
      {
        if ( v13 < 0 )
LABEL_127:
          sub_4261EA(v9, v10, v11);
        v63 = (char *)sub_22077B0(v13);
        memcpy(v63, v12, v13);
        v9 = *(_QWORD *)(a3 + 24);
        v10 = *(_QWORD *)(a3 + 40) - v9;
      }
      else
      {
        v63 = 0;
      }
      if ( v9 )
        j_j___libc_free_0(v9, v10);
      v15 = &v63[v13];
      *(_QWORD *)(a3 + 24) = v63;
      *(_QWORD *)(a3 + 40) = v15;
      goto LABEL_13;
    }
    v14 = *(char **)(a3 + 32);
    if ( v13 <= (unsigned __int64)&v14[-v9] )
    {
      if ( v13 )
      {
        v10 = *(_QWORD *)(v73 + 24);
        memmove((void *)v9, v12, v13);
        v9 = *(_QWORD *)(a3 + 24);
      }
    }
    else
    {
      if ( v14 != (char *)v9 )
      {
        memmove((void *)v9, v12, *(_QWORD *)(a3 + 32) - v9);
        v14 = *(char **)(a3 + 32);
        v11 = *(_QWORD *)(v73 + 32);
        v12 = *(const void **)(v73 + 24);
        v9 = *(_QWORD *)(a3 + 24);
        v5 = (unsigned __int64)&v14[-v9];
      }
      v10 = (__int64)v12 + v5;
      v11 -= (unsigned __int64)v12 + v5;
      if ( v11 )
      {
        v9 = (__int64)v14;
        memmove(v14, (const void *)v10, v11);
        v15 = (char *)(*(_QWORD *)(a3 + 24) + v13);
        goto LABEL_13;
      }
    }
    v15 = (char *)(v9 + v13);
LABEL_13:
    *(_QWORD *)(a3 + 32) = v15;
  }
  v16 = *(__int64 ***)(v73 + 48);
  v17 = *(_QWORD **)(a3 + 48);
  if ( v16 )
  {
    if ( v17 )
    {
      v76 = *(__int64 ***)(v73 + 48);
      v71 = v16 + 9;
      while ( v17 == v76 )
      {
LABEL_30:
        v76 += 3;
        v17 += 3;
        if ( v71 == v76 )
          goto LABEL_31;
      }
      v18 = v76[1];
      v19 = *v76;
      v20 = (__int64 *)*v17;
      v21 = (char *)v18 - (char *)*v76;
      v10 = v17[2] - *v17;
      v74 = v21;
      if ( v10 < (unsigned __int64)v21 )
      {
        if ( v21 )
        {
          v11 = 0x7FFFFFFFFFFFFFF8LL;
          v69 = v76[1];
          if ( (unsigned __int64)v21 > 0x7FFFFFFFFFFFFFF8LL )
            goto LABEL_127;
          v9 = (char *)v76[1] - (char *)*v76;
          v41 = sub_22077B0(v21);
          v18 = v69;
          v42 = (_QWORD *)v41;
        }
        else
        {
          v42 = 0;
        }
        v43 = v42;
        if ( v18 != v19 )
        {
          v70 = v17;
          v44 = v18;
          do
          {
            if ( v43 )
            {
              v11 = v19[1] - *v19;
              *v43 = 0;
              v43[1] = 0;
              v47 = v11;
              v43[2] = 0;
              if ( v11 )
              {
                if ( v11 > 0x7FFFFFFFFFFFFFF0LL )
                  goto LABEL_127;
                v9 = v11;
                v45 = (char *)sub_22077B0(v11);
              }
              else
              {
                v45 = 0;
              }
              v11 = (size_t)&v45[v47];
              *v43 = v45;
              v43[1] = v45;
              v43[2] = &v45[v47];
              v10 = *v19;
              v46 = v19[1] - *v19;
              if ( v19[1] != *v19 )
              {
                v9 = (__int64)v45;
                v45 = (char *)memmove(v45, (const void *)v10, v19[1] - *v19);
              }
              v43[1] = &v45[v46];
            }
            v19 += 3;
            v43 += 3;
          }
          while ( v44 != v19 );
          v17 = v70;
        }
        v48 = (__int64 *)v17[1];
        v49 = (__int64 *)*v17;
        if ( v48 != (__int64 *)*v17 )
        {
          do
          {
            v9 = *v49;
            if ( *v49 )
              j_j___libc_free_0(v9, v49[2] - v9);
            v49 += 3;
          }
          while ( v48 != v49 );
          v49 = (__int64 *)*v17;
        }
        if ( v49 )
        {
          v9 = (__int64)v49;
          j_j___libc_free_0(v49, v17[2] - (_QWORD)v49);
        }
        *v17 = v42;
        v27 = (char *)v42 + v74;
        v17[2] = v27;
        goto LABEL_29;
      }
      v22 = (char *)v17[1];
      v10 = v22 - (char *)v20;
      if ( v21 > (unsigned __int64)(v22 - (char *)v20) )
      {
        v34 = 0xAAAAAAAAAAAAAAABLL * ((v22 - (char *)v20) >> 3);
        if ( v10 > 0 )
        {
          do
          {
            v35 = (char **)v19;
            v9 = (__int64)v20;
            v19 += 3;
            v20 += 3;
            sub_ED6D60(v9, v35);
            --v34;
          }
          while ( v34 );
          v22 = (char *)v17[1];
          v20 = (__int64 *)*v17;
          v18 = v76[1];
          v19 = *v76;
          v10 = (__int64)&v22[-*v17];
        }
        v36 = (__int64 *)((char *)v19 + v10);
        v27 = (char *)v20 + v74;
        if ( (__int64 *)((char *)v19 + v10) == v18 )
          goto LABEL_29;
        v37 = v18;
        do
        {
          if ( v22 )
          {
            v40 = v36[1] - *v36;
            *(_QWORD *)v22 = 0;
            *((_QWORD *)v22 + 1) = 0;
            *((_QWORD *)v22 + 2) = 0;
            if ( v40 )
            {
              if ( v40 > 0x7FFFFFFFFFFFFFF0LL )
                goto LABEL_127;
              v9 = v40;
              v38 = (char *)sub_22077B0(v40);
            }
            else
            {
              v38 = 0;
            }
            *(_QWORD *)v22 = v38;
            *((_QWORD *)v22 + 2) = &v38[v40];
            *((_QWORD *)v22 + 1) = v38;
            v10 = *v36;
            v39 = v36[1] - *v36;
            if ( v36[1] != *v36 )
            {
              v9 = (__int64)v38;
              v38 = (char *)memmove(v38, (const void *)v10, v36[1] - *v36);
            }
            *((_QWORD *)v22 + 1) = &v38[v39];
          }
          v36 += 3;
          v22 += 24;
        }
        while ( v36 != v37 );
      }
      else
      {
        if ( v21 <= 0 )
          goto LABEL_27;
        v68 = v17;
        v23 = (char **)*v76;
        v24 = v20;
        v25 = 0xAAAAAAAAAAAAAAABLL * (v21 >> 3);
        do
        {
          v26 = v23;
          v9 = (__int64)v24;
          v23 += 3;
          v24 += 3;
          sub_ED6D60(v9, v26);
          --v25;
        }
        while ( v25 );
        v17 = v68;
        v11 = v74;
        v20 = (__int64 *)((char *)v20 + v74);
        while ( v22 != (char *)v20 )
        {
          v9 = *v20;
          if ( *v20 )
            j_j___libc_free_0(v9, v20[2] - v9);
          v20 += 3;
LABEL_27:
          ;
        }
      }
      v27 = (char *)(*v17 + v74);
LABEL_29:
      v17[1] = v27;
      goto LABEL_30;
    }
    v9 = 72;
    v72 = sub_22077B0(72);
    if ( v72 )
    {
      v50 = (_QWORD *)v72;
      do
      {
        v51 = (char *)v16[1] - (char *)*v16;
        *v50 = 0;
        v50[1] = 0;
        v50[2] = 0;
        if ( v51 )
        {
          if ( v51 > 0x7FFFFFFFFFFFFFF8LL )
            goto LABEL_127;
          v9 = v51;
          v52 = (_QWORD *)sub_22077B0(v51);
        }
        else
        {
          v52 = 0;
        }
        *v50 = v52;
        v50[1] = v52;
        v50[2] = (char *)v52 + v51;
        v53 = v16[1];
        if ( v53 != *v16 )
        {
          v77 = v50;
          v54 = *v16;
          do
          {
            if ( v52 )
            {
              v11 = v54[1] - *v54;
              *v52 = 0;
              v52[1] = 0;
              v57 = v11;
              v52[2] = 0;
              if ( v11 )
              {
                if ( v11 > 0x7FFFFFFFFFFFFFF0LL )
                  goto LABEL_127;
                v9 = v11;
                v55 = (char *)sub_22077B0(v11);
              }
              else
              {
                v55 = 0;
              }
              v11 = (size_t)&v55[v57];
              *v52 = v55;
              v52[1] = v55;
              v52[2] = &v55[v57];
              v10 = *v54;
              v56 = v54[1] - *v54;
              if ( v54[1] != *v54 )
              {
                v9 = (__int64)v55;
                v55 = (char *)memmove(v55, (const void *)v10, v54[1] - *v54);
              }
              v52[1] = &v55[v56];
            }
            v54 += 3;
            v52 += 3;
          }
          while ( v53 != v54 );
          v50 = v77;
        }
        v50[1] = v52;
        v16 += 3;
        v50 += 3;
      }
      while ( (_QWORD *)(v72 + 72) != v50 );
    }
    v58 = *(_QWORD **)(a3 + 48);
    *(_QWORD *)(a3 + 48) = v72;
    v59 = v58 + 9;
    if ( v58 )
    {
      do
      {
        v60 = (_QWORD *)*(v59 - 3);
        v61 = (_QWORD *)*(v59 - 2);
        v59 -= 3;
        v62 = v60;
        if ( v61 != v60 )
        {
          do
          {
            if ( *v62 )
              j_j___libc_free_0(*v62, v62[2] - *v62);
            v62 += 3;
          }
          while ( v61 != v62 );
          v60 = (_QWORD *)*v59;
        }
        if ( v60 )
          j_j___libc_free_0(v60, v59[2] - (_QWORD)v60);
      }
      while ( v58 != v59 );
      j_j___libc_free_0(v58, 72);
    }
  }
  else
  {
    *(_QWORD *)(a3 + 48) = 0;
    v64 = v17 + 9;
    if ( v17 )
    {
      do
      {
        v65 = (_QWORD *)*(v64 - 3);
        v66 = (_QWORD *)*(v64 - 2);
        v64 -= 3;
        v67 = v65;
        if ( v66 != v65 )
        {
          do
          {
            if ( *v67 )
              j_j___libc_free_0(*v67, v67[2] - *v67);
            v67 += 3;
          }
          while ( v66 != v67 );
          v65 = (_QWORD *)*v64;
        }
        if ( v65 )
          j_j___libc_free_0(v65, v64[2] - (_QWORD)v65);
      }
      while ( v17 != v64 );
      j_j___libc_free_0(v17, 72);
    }
  }
LABEL_31:
  *(__m128i *)(a3 + 56) = _mm_loadu_si128((const __m128i *)(v73 + 56));
  *(_QWORD *)(a3 + 72) = *(_QWORD *)(v73 + 72);
  if ( *(unsigned int *)(a2 + 488) >= v82 )
  {
    v28 = *(_QWORD **)(a2 + 128);
    v29 = *(void (**)(void))(*v28 + 32LL);
    if ( (char *)v29 == (char *)sub_ED7380 )
    {
      v30 = v28[3];
      v31 = (unsigned __int16 *)v28[2];
      if ( !v30 )
      {
        v30 = *v31++;
        v28[3] = v30;
      }
      v28[2] = v31 + 4;
      v32 = *((_QWORD *)v31 + 1);
      v28[2] = v31 + 8;
      v33 = (__int64)v31 + *((_QWORD *)v31 + 2) + v32 + 24;
      --v28[4];
      v28[2] = v33;
      v28[3] = v30 - 1;
    }
    else
    {
      v29();
    }
    *(_DWORD *)(a2 + 488) = 0;
  }
  v6 = a2;
  sub_ED8620(a1, a2);
LABEL_3:
  if ( (v80 & 1) != 0 || (v80 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    sub_C63C30(&v80, v6);
  return a1;
}
