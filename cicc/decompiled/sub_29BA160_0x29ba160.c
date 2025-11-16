// Function: sub_29BA160
// Address: 0x29ba160
//
__int64 __fastcall sub_29BA160(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5)
{
  _QWORD **v5; // r14
  _QWORD **v9; // rcx
  _QWORD **v10; // rdx
  __int64 v11; // rax
  unsigned __int64 v12; // rdi
  unsigned __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  _QWORD *v16; // rax
  _QWORD *v17; // rdx
  __int64 v18; // rdi
  char *v19; // r9
  char *v20; // rdx
  _QWORD *v21; // rdi
  __int64 v22; // rax
  __int64 v23; // r8
  __int64 v24; // rcx
  __int64 v25; // rsi
  __int64 *v26; // rdx
  __int64 v27; // rsi
  __int64 *v28; // rdx
  __int64 v29; // rcx
  __int64 *v30; // r10
  double v31; // xmm3_8
  _QWORD *v32; // r9
  _QWORD *v33; // r11
  char *v34; // r14
  char *v35; // rcx
  __int64 v36; // rax
  __int64 v37; // rdx
  char *v38; // rax
  char *v39; // rax
  char *v40; // rdx
  char *v41; // rax
  char *v42; // rcx
  __int64 result; // rax
  __int64 i; // rcx
  __int64 v45; // rdx
  _QWORD **v46; // [rsp+0h] [rbp-80h] BYREF
  _QWORD **v47; // [rsp+8h] [rbp-78h]
  __int64 v48; // [rsp+10h] [rbp-70h] BYREF
  char *v49[5]; // [rsp+20h] [rbp-60h] BYREF
  __int64 v50; // [rsp+48h] [rbp-38h]

  v5 = &v46;
  sub_29B78C0((__int64 *)v49, (__int64 *)(a2 + 32), (__int64 *)(a3 + 32), a4, a5);
  sub_29B92F0((__int64)&v46, v49);
  v9 = v46;
  v10 = v47;
  v46 = 0;
  v11 = v48;
  v12 = *(_QWORD *)(a2 + 32);
  v47 = 0;
  *(_QWORD *)(a2 + 32) = v9;
  *(_QWORD *)(a2 + 40) = v10;
  *(_QWORD *)(a2 + 48) = v11;
  v48 = 0;
  if ( v12 )
  {
    j_j___libc_free_0(v12);
    v12 = (unsigned __int64)v46;
    v9 = *(_QWORD ***)(a2 + 32);
    v10 = *(_QWORD ***)(a2 + 40);
  }
  *(double *)(a2 + 16) = *(double *)(a2 + 16) + *(double *)(a3 + 16);
  *(_QWORD *)(a2 + 24) += *(_QWORD *)(a3 + 24);
  *(_QWORD *)a2 = **v9;
  v13 = 0;
  if ( v9 != v10 )
  {
    do
    {
      v9[v13][4] = a2;
      *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 32) + 8 * v13) + 8LL) = v13;
      v9 = *(_QWORD ***)(a2 + 32);
      ++v13;
    }
    while ( (__int64)(*(_QWORD *)(a2 + 40) - (_QWORD)v9) >> 3 > v13 );
  }
  if ( v12 )
    j_j___libc_free_0(v12);
  sub_29B9D00((unsigned __int64 *)a2, (unsigned __int64 *)a3);
  v14 = *(_QWORD *)(a3 + 32);
  if ( v14 != *(_QWORD *)(a3 + 40) )
    *(_QWORD *)(a3 + 40) = v14;
  v15 = *(_QWORD *)(a3 + 56);
  if ( v15 != *(_QWORD *)(a3 + 64) )
    *(_QWORD *)(a3 + 64) = v15;
  v16 = *(_QWORD **)(a2 + 56);
  v17 = *(_QWORD **)(a2 + 64);
  if ( v17 != v16 )
  {
    while ( a2 != *v16 )
    {
      v16 += 2;
      if ( v17 == v16 )
        goto LABEL_30;
    }
    v18 = v16[1];
    if ( v18 )
    {
      v19 = *(char **)(a2 + 40);
      v20 = *(char **)(a2 + 32);
      v21 = (_QWORD *)(v18 + 16);
      v22 = 0;
      v23 = qword_5007B18;
      v24 = qword_5007B10;
      v46 = (_QWORD **)v21;
      v49[0] = v20;
      v49[1] = v19;
      v49[2] = (char *)qword_5007B10;
      v49[3] = (char *)qword_5007B18;
      v49[4] = (char *)qword_5007B10;
      v50 = qword_5007B18;
      v47 = 0;
      if ( v20 == v19 )
      {
        if ( qword_5007B10 == qword_5007B18 )
        {
LABEL_23:
          v30 = &v48;
          v31 = 0.0;
          if ( v21 )
            goto LABEL_26;
LABEL_24:
          while ( v30 != (__int64 *)++v5 )
          {
            while ( 1 )
            {
              v21 = *v5;
              if ( !*v5 )
                break;
LABEL_26:
              v32 = (_QWORD *)*v21;
              if ( *v21 == v21[1] )
                goto LABEL_24;
              do
                v31 = v31
                    + sub_29B8240(
                        *(_QWORD *)(*(_QWORD *)*v32 + 40LL),
                        *(_QWORD *)(*(_QWORD *)*v32 + 16LL),
                        *(_QWORD *)(*(_QWORD *)(*v32 + 8LL) + 40LL),
                        *(_QWORD *)(*v32 + 16LL),
                        *(_BYTE *)(*v32 + 24LL));
              while ( v33 != v32 );
              if ( v30 == (__int64 *)++v5 )
                goto LABEL_29;
            }
          }
LABEL_29:
          *(double *)(a2 + 8) = v31;
          goto LABEL_30;
        }
      }
      else
      {
        do
        {
          v25 = *(_QWORD *)v20;
          v20 += 8;
          *(_QWORD *)(v25 + 40) = v22;
          v22 += *(_QWORD *)(v25 + 16);
        }
        while ( v19 != v20 );
        if ( v24 == v23 )
        {
LABEL_21:
          v28 = (__int64 *)v24;
          if ( v24 != v50 )
          {
            do
            {
              v29 = *v28++;
              *(_QWORD *)(v29 + 40) = v22;
              v22 += *(_QWORD *)(v29 + 16);
            }
            while ( (__int64 *)v50 != v28 );
          }
          goto LABEL_23;
        }
      }
      v26 = (__int64 *)v24;
      do
      {
        v27 = *v26++;
        *(_QWORD *)(v27 + 40) = v22;
        v22 += *(_QWORD *)(v27 + 16);
      }
      while ( (__int64 *)v23 != v26 );
      goto LABEL_21;
    }
  }
LABEL_30:
  v34 = *(char **)(a1 + 160);
  v35 = *(char **)(a1 + 152);
  v36 = (v34 - v35) >> 5;
  v37 = (v34 - v35) >> 3;
  if ( v36 > 0 )
  {
    v38 = &v35[32 * v36];
    while ( a3 != *(_QWORD *)v35 )
    {
      if ( a3 == *((_QWORD *)v35 + 1) )
      {
        v35 += 8;
        goto LABEL_37;
      }
      if ( a3 == *((_QWORD *)v35 + 2) )
      {
        v35 += 16;
        goto LABEL_37;
      }
      if ( a3 == *((_QWORD *)v35 + 3) )
      {
        v35 += 24;
        goto LABEL_37;
      }
      v35 += 32;
      if ( v38 == v35 )
      {
        v37 = (v34 - v35) >> 3;
        goto LABEL_51;
      }
    }
    goto LABEL_37;
  }
LABEL_51:
  if ( v37 == 2 )
  {
LABEL_58:
    if ( a3 != *(_QWORD *)v35 )
    {
      v35 += 8;
      goto LABEL_54;
    }
    goto LABEL_37;
  }
  if ( v37 != 3 )
  {
    if ( v37 != 1 )
      goto LABEL_47;
LABEL_54:
    if ( a3 != *(_QWORD *)v35 )
      goto LABEL_47;
    goto LABEL_37;
  }
  if ( a3 != *(_QWORD *)v35 )
  {
    v35 += 8;
    goto LABEL_58;
  }
LABEL_37:
  if ( v34 != v35 )
  {
    v39 = v35 + 8;
    if ( v34 == v35 + 8 )
      goto LABEL_43;
    do
    {
      if ( a3 != *(_QWORD *)v39 )
      {
        *(_QWORD *)v35 = *(_QWORD *)v39;
        v35 += 8;
      }
      v39 += 8;
    }
    while ( v34 != v39 );
    if ( v34 != v35 )
    {
LABEL_43:
      v40 = *(char **)(a1 + 160);
      if ( v34 != v40 )
      {
        v41 = (char *)memmove(v35, v34, v40 - v34);
        v40 = *(char **)(a1 + 160);
        v35 = v41;
      }
      v42 = &v35[v40 - v34];
      if ( v40 != v42 )
        *(_QWORD *)(a1 + 160) = v42;
    }
  }
LABEL_47:
  result = *(_QWORD *)(a2 + 56);
  for ( i = *(_QWORD *)(a2 + 64); i != result; *(_WORD *)(v45 + 112) = 0 )
  {
    v45 = *(_QWORD *)(result + 8);
    result += 16;
  }
  return result;
}
