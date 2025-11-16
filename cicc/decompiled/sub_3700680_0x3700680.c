// Function: sub_3700680
// Address: 0x3700680
//
void __fastcall sub_3700680(_QWORD *a1, char *a2, unsigned __int64 a3, __int64 a4, char a5, __int64 a6)
{
  __int64 *v6; // r15
  _QWORD *v7; // r14
  unsigned __int64 v9; // r12
  __int64 v10; // r13
  __int64 v11; // r12
  __int64 v12; // r15
  __int64 v13; // rax
  unsigned __int64 v14; // rdx
  char *v15; // rdi
  char *v16; // rax
  _BYTE *v17; // r11
  size_t v18; // r9
  __int64 v19; // rax
  char *v20; // r15
  size_t v21; // r12
  unsigned __int64 v22; // r12
  _QWORD *v23; // rcx
  const void *v24; // rsi
  __int64 v25; // r12
  __int64 v26; // rax
  __int64 v27; // rax
  _QWORD *v28; // rax
  _BYTE *v29; // r13
  size_t v30; // r12
  __int64 v31; // rax
  char *v32; // rax
  unsigned __int64 v33; // rbx
  __int64 v34; // rax
  char *v35; // rcx
  size_t v36; // r12
  _BYTE *v38; // [rsp+8h] [rbp-B8h]
  unsigned __int64 v40; // [rsp+18h] [rbp-A8h]
  unsigned __int64 v41; // [rsp+18h] [rbp-A8h]
  size_t v42; // [rsp+18h] [rbp-A8h]
  size_t v43; // [rsp+28h] [rbp-98h] BYREF
  unsigned __int64 v44[2]; // [rsp+30h] [rbp-90h] BYREF
  _QWORD v45[3]; // [rsp+40h] [rbp-80h] BYREF
  unsigned __int64 v46; // [rsp+58h] [rbp-68h]
  char *v47; // [rsp+60h] [rbp-60h]
  char *v48; // [rsp+68h] [rbp-58h]
  __int64 v49; // [rsp+70h] [rbp-50h]
  __int64 v50; // [rsp+78h] [rbp-48h]
  char v51; // [rsp+80h] [rbp-40h]

  v6 = (__int64 *)a3;
  v7 = a1;
  *a1 = *(_QWORD *)a2;
  *(_QWORD *)a2 = 0;
  v9 = *(_QWORD *)(a3 + 8) - *(_QWORD *)a3;
  a1[1] = 0;
  a1[2] = 0;
  a1[3] = 0;
  if ( v9 )
  {
    if ( v9 > 0x7FFFFFFFFFFFFFD0LL )
      goto LABEL_63;
    v10 = sub_22077B0(v9);
  }
  else
  {
    v9 = 0;
    v10 = 0;
  }
  a1[1] = v10;
  a1[2] = v10;
  a1[3] = v10 + v9;
  v11 = v6[1];
  if ( v11 != *v6 )
  {
    v12 = *v6;
    while ( !v10 )
    {
LABEL_11:
      v12 += 80;
      v10 += 80;
      if ( v11 == v12 )
        goto LABEL_24;
    }
    a1 = (_QWORD *)(v10 + 16);
    *(_QWORD *)v10 = v10 + 16;
    v17 = *(_BYTE **)v12;
    v18 = *(_QWORD *)(v12 + 8);
    if ( v18 + *(_QWORD *)v12 && !v17 )
      goto LABEL_64;
    v44[0] = *(_QWORD *)(v12 + 8);
    if ( v18 > 0xF )
    {
      v38 = v17;
      v42 = v18;
      v19 = sub_22409D0(v10, v44, 0);
      v18 = v42;
      v17 = v38;
      *(_QWORD *)v10 = v19;
      a1 = (_QWORD *)v19;
      *(_QWORD *)(v10 + 16) = v44[0];
    }
    else
    {
      if ( v18 == 1 )
      {
        *(_BYTE *)(v10 + 16) = *v17;
LABEL_18:
        *(_QWORD *)(v10 + 8) = v18;
        *((_BYTE *)a1 + v18) = 0;
        *(_DWORD *)(v10 + 32) = *(_DWORD *)(v12 + 32);
        *(_DWORD *)(v10 + 36) = *(_DWORD *)(v12 + 36);
        a3 = *(_QWORD *)(v12 + 48) - *(_QWORD *)(v12 + 40);
        *(_QWORD *)(v10 + 40) = 0;
        *(_QWORD *)(v10 + 48) = 0;
        *(_QWORD *)(v10 + 56) = 0;
        if ( a3 )
        {
          if ( a3 > 0x7FFFFFFFFFFFFFF8LL )
            goto LABEL_63;
          v40 = a3;
          v13 = sub_22077B0(a3);
          v14 = v40;
          v15 = (char *)v13;
        }
        else
        {
          v14 = 0;
          v15 = 0;
        }
        *(_QWORD *)(v10 + 40) = v15;
        *(_QWORD *)(v10 + 56) = &v15[v14];
        *(_QWORD *)(v10 + 48) = v15;
        a2 = *(char **)(v12 + 40);
        a3 = *(_QWORD *)(v12 + 48) - (_QWORD)a2;
        if ( *(char **)(v12 + 48) != a2 )
        {
          v41 = *(_QWORD *)(v12 + 48) - (_QWORD)a2;
          v16 = (char *)memmove(v15, a2, a3);
          a3 = v41;
          v15 = v16;
        }
        *(_QWORD *)(v10 + 48) = &v15[a3];
        *(_QWORD *)(v10 + 64) = *(_QWORD *)(v12 + 64);
        *(_QWORD *)(v10 + 72) = *(_QWORD *)(v12 + 72);
        goto LABEL_11;
      }
      if ( !v18 )
        goto LABEL_18;
    }
    a2 = v17;
    memcpy(a1, v17, v18);
    v18 = v44[0];
    a1 = *(_QWORD **)v10;
    goto LABEL_18;
  }
LABEL_24:
  a1 = v7 + 6;
  v7[2] = v10;
  v7[4] = v7 + 6;
  v20 = *(char **)a4;
  v21 = *(_QWORD *)(a4 + 8);
  if ( v21 + *(_QWORD *)a4 && !v20 )
    goto LABEL_64;
  v44[0] = *(_QWORD *)(a4 + 8);
  if ( v21 > 0xF )
  {
    v27 = sub_22409D0((__int64)(v7 + 4), v44, 0);
    v7[4] = v27;
    a1 = (_QWORD *)v27;
    v7[6] = v44[0];
LABEL_42:
    a2 = v20;
    memcpy(a1, v20, v21);
    v21 = v44[0];
    a1 = (_QWORD *)v7[4];
    goto LABEL_29;
  }
  if ( v21 == 1 )
  {
    *((_BYTE *)v7 + 48) = *v20;
    goto LABEL_29;
  }
  if ( v21 )
    goto LABEL_42;
LABEL_29:
  v7[5] = v21;
  *((_BYTE *)a1 + v21) = 0;
  v7[8] = *(_QWORD *)(a4 + 32);
  v22 = *(_QWORD *)(a4 + 48) - *(_QWORD *)(a4 + 40);
  v7[9] = 0;
  v7[10] = 0;
  v7[11] = 0;
  if ( v22 )
  {
    if ( v22 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_63;
    a1 = (_QWORD *)v22;
    v23 = (_QWORD *)sub_22077B0(v22);
  }
  else
  {
    v22 = 0;
    v23 = 0;
  }
  v7[9] = v23;
  v7[11] = (char *)v23 + v22;
  v7[10] = v23;
  v24 = *(const void **)(a4 + 40);
  v25 = *(_QWORD *)(a4 + 48) - (_QWORD)v24;
  if ( *(const void **)(a4 + 48) != v24 )
  {
    a1 = v23;
    v23 = memmove(v23, v24, *(_QWORD *)(a4 + 48) - (_QWORD)v24);
  }
  v51 = 0;
  v7[10] = (char *)v23 + v25;
  v7[12] = *(_QWORD *)(a4 + 64);
  v26 = *(_QWORD *)(a4 + 72);
  v7[15] = 0;
  v7[13] = v26;
  v7[16] = 0;
  *((_BYTE *)v7 + 112) = a5;
  v7[17] = 0x1000000000LL;
  v7[18] = v7 + 20;
  v7[19] = 0;
  *((_BYTE *)v7 + 160) = 0;
  if ( *(_BYTE *)(a6 + 80) )
  {
    v28 = v45;
    v44[0] = (unsigned __int64)v45;
    v29 = *(_BYTE **)a6;
    v30 = *(_QWORD *)(a6 + 8);
    if ( !(v30 + *(_QWORD *)a6) || v29 )
    {
      v43 = *(_QWORD *)(a6 + 8);
      if ( v30 > 0xF )
      {
        v44[0] = sub_22409D0((__int64)v44, &v43, 0);
        a1 = (_QWORD *)v44[0];
        v45[0] = v43;
      }
      else
      {
        if ( v30 == 1 )
        {
          a3 = (unsigned __int8)*v29;
          LOBYTE(v45[0]) = *v29;
LABEL_52:
          v44[1] = v30;
          *((_BYTE *)v28 + v30) = 0;
          v31 = *(_QWORD *)(a6 + 32);
          a2 = *(char **)(a6 + 40);
          v46 = 0;
          v47 = 0;
          v45[2] = v31;
          v32 = *(char **)(a6 + 48);
          v48 = 0;
          v33 = v32 - a2;
          if ( v32 == a2 )
          {
            v36 = 0;
            v33 = 0;
            v35 = 0;
            goto LABEL_55;
          }
          if ( v33 <= 0x7FFFFFFFFFFFFFF8LL )
          {
            v34 = sub_22077B0(v33);
            a2 = *(char **)(a6 + 40);
            v35 = (char *)v34;
            v32 = *(char **)(a6 + 48);
            v36 = v32 - a2;
LABEL_55:
            v46 = (unsigned __int64)v35;
            v47 = v35;
            v48 = &v35[v33];
            if ( v32 != a2 )
              v35 = (char *)memmove(v35, a2, v36);
            v51 = 1;
            v47 = &v35[v36];
            v49 = *(_QWORD *)(a6 + 64);
            v50 = *(_QWORD *)(a6 + 72);
            goto LABEL_35;
          }
LABEL_63:
          sub_4261EA(a1, a2, a3);
        }
        if ( !v30 )
          goto LABEL_52;
        a1 = v45;
      }
      memcpy(a1, v29, v30);
      v30 = v43;
      v28 = (_QWORD *)v44[0];
      goto LABEL_52;
    }
LABEL_64:
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  }
LABEL_35:
  sub_36FFCE0(v7, (__int64)v44);
  if ( v51 )
  {
    v51 = 0;
    if ( v46 )
      j_j___libc_free_0(v46);
    if ( (_QWORD *)v44[0] != v45 )
      j_j___libc_free_0(v44[0]);
  }
}
