// Function: sub_24A91D0
// Address: 0x24a91d0
//
__int64 __fastcall sub_24A91D0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *v4; // r15
  __int64 v5; // rbx
  double v6; // xmm0_8
  double v7; // xmm0_8
  __int64 v8; // r13
  __int64 v9; // rcx
  __int64 v10; // rax
  unsigned int v11; // esi
  __int64 *v12; // rdx
  __int64 v13; // r8
  __int64 v14; // rax
  __int64 v15; // rcx
  __int64 v16; // rdx
  __int64 v17; // rdx
  unsigned int v18; // r12d
  __int64 *v19; // rax
  __int64 v20; // r8
  __int64 v21; // rax
  __int64 v22; // rdx
  _QWORD *v23; // rdi
  _QWORD *v24; // r12
  _QWORD *i; // r13
  void **v26; // rax
  int v28; // eax
  int v29; // edx
  int v30; // r9d
  int v31; // eax
  __int64 v32; // rdx
  __int64 v33; // rax
  double v34; // xmm0_8
  double v35; // xmm0_8
  unsigned __int64 v36; // rsi
  __int64 v37; // rax
  void **k; // rbx
  __int64 v39; // rax
  _QWORD *j; // rbx
  __int64 v41; // rdx
  int v42; // edi
  const char *v43; // [rsp+18h] [rbp-108h]
  const char *v44; // [rsp+28h] [rbp-F8h]
  __int64 v45; // [rsp+40h] [rbp-E0h]
  __int64 *v46; // [rsp+48h] [rbp-D8h]
  double v47; // [rsp+48h] [rbp-D8h]
  __int64 v48; // [rsp+58h] [rbp-C8h] BYREF
  __int64 v49; // [rsp+60h] [rbp-C0h]
  __int64 v50; // [rsp+68h] [rbp-B8h]
  __int64 *v51; // [rsp+70h] [rbp-B0h] BYREF
  void **v52; // [rsp+78h] [rbp-A8h]
  __int64 *v53; // [rsp+90h] [rbp-90h] BYREF
  _QWORD *v54; // [rsp+98h] [rbp-88h]
  __int64 *v55; // [rsp+B0h] [rbp-70h] BYREF
  _QWORD *v56; // [rsp+B8h] [rbp-68h]
  __int64 *v57[2]; // [rsp+D0h] [rbp-50h] BYREF
  __int64 v58; // [rsp+E0h] [rbp-40h]

  v43 = *(const char **)a1;
  sub_FE7FB0(&v48, *(const char **)a1, a3, a2);
  v46 = (__int64 *)sub_C33320();
  v4 = (__int64 *)sub_C33340();
  if ( v46 == v4 )
    sub_C3C500(&v51, (__int64)v46);
  else
    sub_C373C0(&v51, (__int64)v46);
  if ( v51 == v4 )
    sub_C3CEB0((void **)&v51, 0);
  else
    sub_C37310((__int64)&v51, 0);
  if ( v46 == v4 )
    sub_C3C500(&v53, (__int64)v46);
  else
    sub_C373C0(&v53, (__int64)v46);
  if ( v4 == v53 )
    sub_C3CEB0((void **)&v53, 0);
  else
    sub_C37310((__int64)&v53, 0);
  v5 = *((_QWORD *)v43 + 10);
  v44 = v43 + 72;
  if ( (const char *)v5 != v43 + 72 )
  {
    do
    {
      while ( 1 )
      {
        v8 = v5 - 24;
        v9 = *(_QWORD *)(a1 + 280);
        if ( !v5 )
          v8 = 0;
        v10 = *(unsigned int *)(a1 + 296);
        if ( (_DWORD)v10 )
        {
          v11 = (v10 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
          v12 = (__int64 *)(v9 + 16LL * v11);
          v13 = *v12;
          if ( v8 != *v12 )
          {
            v29 = 1;
            while ( v13 != -4096 )
            {
              v30 = v29 + 1;
              v11 = (v10 - 1) & (v29 + v11);
              v12 = (__int64 *)(v9 + 16LL * v11);
              v13 = *v12;
              if ( v8 == *v12 )
                goto LABEL_27;
              v29 = v30;
            }
            goto LABEL_22;
          }
LABEL_27:
          if ( v12 != (__int64 *)(v9 + 16 * v10) && v12[1] )
            break;
        }
LABEL_22:
        v5 = *(_QWORD *)(v5 + 8);
        if ( v44 == (const char *)v5 )
          goto LABEL_44;
      }
      v14 = sub_FDD2C0(&v48, v8, 0);
      v15 = *(_QWORD *)(a1 + 280);
      v50 = v16;
      v17 = *(unsigned int *)(a1 + 296);
      v49 = v14;
      if ( (_DWORD)v17 )
      {
        v18 = (v17 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
        v19 = (__int64 *)(v15 + 16LL * v18);
        v20 = *v19;
        if ( v8 == *v19 )
          goto LABEL_31;
        v28 = 1;
        while ( v20 != -4096 )
        {
          v42 = v28 + 1;
          v18 = (v17 - 1) & (v28 + v18);
          v19 = (__int64 *)(v15 + 16LL * v18);
          v20 = *v19;
          if ( v8 == *v19 )
            goto LABEL_31;
          v28 = v42;
        }
      }
      v19 = (__int64 *)(v15 + 16 * v17);
LABEL_31:
      v21 = *(_QWORD *)(v19[1] + 16);
      v45 = v49;
      if ( v21 >= 0 )
        v6 = (double)(int)v21;
      else
        v6 = (double)(int)(v21 & 1 | ((unsigned __int64)v21 >> 1))
           + (double)(int)(v21 & 1 | ((unsigned __int64)v21 >> 1));
      sub_C3B1B0((__int64)v57, v6);
      sub_C407B0(&v55, (__int64 *)v57, v46);
      sub_C338F0((__int64)v57);
      if ( v4 == v51 )
        sub_C3D800((__int64 *)&v51, (__int64)&v55, 1u);
      else
        sub_C3ADF0((__int64)&v51, (__int64)&v55, 1);
      if ( v4 == v55 )
      {
        if ( v56 )
        {
          v22 = *(v56 - 1);
          v23 = &v56[3 * v22];
          if ( v56 != v23 )
          {
            v24 = &v56[3 * v22];
            do
            {
              v24 -= 3;
              sub_91D830(v24);
            }
            while ( v56 != v24 );
            v23 = v24;
          }
          j_j_j___libc_free_0_0((unsigned __int64)(v23 - 1));
        }
      }
      else
      {
        sub_C338F0((__int64)&v55);
      }
      if ( v45 < 0 )
        v7 = (double)(int)(v45 & 1 | ((unsigned __int64)v45 >> 1))
           + (double)(int)(v45 & 1 | ((unsigned __int64)v45 >> 1));
      else
        v7 = (double)(int)v45;
      sub_C3B1B0((__int64)v57, v7);
      sub_C407B0(&v55, (__int64 *)v57, v46);
      sub_C338F0((__int64)v57);
      if ( v4 == v53 )
        sub_C3D800((__int64 *)&v53, (__int64)&v55, 1u);
      else
        sub_C3ADF0((__int64)&v53, (__int64)&v55, 1);
      if ( v4 != v55 )
      {
        sub_C338F0((__int64)&v55);
        goto LABEL_22;
      }
      if ( !v56 )
        goto LABEL_22;
      for ( i = &v56[3 * *(v56 - 1)]; v56 != i; sub_91D830(i) )
        i -= 3;
      j_j_j___libc_free_0_0((unsigned __int64)(i - 1));
      v5 = *(_QWORD *)(v5 + 8);
    }
    while ( v44 != (const char *)v5 );
  }
LABEL_44:
  v26 = (void **)&v51;
  if ( v4 == v51 )
    v26 = v52;
  if ( (*((_BYTE *)v26 + 20) & 7) == 3 )
    goto LABEL_47;
  v31 = v4 == v53 ? sub_C3E510((__int64)&v53, (__int64)&v51) : sub_C37950((__int64)&v53, (__int64)&v51);
  if ( v31 == 1 )
    goto LABEL_47;
  if ( v4 == v51 )
    sub_C3C790(v57, &v51);
  else
    sub_C33EB0(v57, (__int64 *)&v51);
  if ( v4 == v57[0] )
    sub_C3EF50(v57, (__int64)&v53, 1u);
  else
    sub_C3B6C0((__int64)v57, (__int64)&v53, 1);
  v47 = sub_C41B00((__int64 *)v57);
  sub_91D830(v57);
  if ( v47 < 1.001 && v47 > 0.999 )
  {
LABEL_47:
    if ( v4 == v53 )
    {
      if ( v54 )
      {
        v39 = 3LL * *(v54 - 1);
        for ( j = &v54[v39]; v54 != j; sub_91D830(j) )
          j -= 3;
        j_j_j___libc_free_0_0((unsigned __int64)(j - 1));
      }
    }
    else
    {
      sub_C338F0((__int64)&v53);
    }
    if ( v4 == v51 )
    {
      if ( v52 )
      {
        v37 = 3LL * (_QWORD)*(v52 - 1);
        for ( k = &v52[v37]; v52 != k; sub_91D830(k) )
          k -= 3;
        j_j_j___libc_free_0_0((unsigned __int64)(k - 1));
      }
    }
    else
    {
      sub_C338F0((__int64)&v51);
    }
    return sub_FDC110(&v48);
  }
  else
  {
    v32 = *((_QWORD *)v43 + 10);
    if ( v32 )
      v32 -= 24;
    sub_24A2C40(v57, (__int64 *)(a1 + 272), v32);
    v33 = *(_QWORD *)(*(_QWORD *)(v58 + 8) + 16LL);
    if ( v33 < 0 )
    {
      v41 = *(_QWORD *)(*(_QWORD *)(v58 + 8) + 16LL) & 1LL | ((unsigned __int64)v33 >> 1);
      v34 = (double)(int)v41 + (double)(int)v41;
    }
    else
    {
      v34 = (double)(int)v33;
    }
    v35 = v34 * v47 + 0.5;
    if ( v35 >= 9.223372036854776e18 )
      v36 = (unsigned int)(int)(v35 - 9.223372036854776e18) ^ 0x8000000000000000LL;
    else
      v36 = (unsigned int)(int)v35;
    if ( !v36 )
      v36 = 1;
    if ( v36 != v33 )
      sub_B2F4C0((__int64)v43, v36, 0, 0);
    sub_91D830(&v53);
    sub_91D830(&v51);
    return sub_FDC110(&v48);
  }
}
