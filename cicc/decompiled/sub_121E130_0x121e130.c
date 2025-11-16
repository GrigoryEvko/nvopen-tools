// Function: sub_121E130
// Address: 0x121e130
//
__int64 __fastcall sub_121E130(__int64 a1)
{
  _DWORD *v2; // r12
  int v3; // eax
  __int64 v4; // r12
  __int64 v5; // r14
  __int64 v6; // rsi
  __int64 v7; // r15
  _BYTE *v8; // rsi
  __int64 v9; // rax
  __int64 v10; // rcx
  __int64 v11; // rdi
  __int64 v12; // rsi
  _QWORD *v13; // r12
  unsigned __int64 v14; // rdx
  __int64 v15; // rdi
  unsigned int v16; // r12d
  __int64 v18; // r13
  __int64 v19; // rdi
  __int64 v20; // rdi
  __int64 v21; // rdi
  __int64 v22; // rdi
  void **i; // r13
  __int64 v24; // rdi
  const char *v25; // rax
  unsigned __int64 v26; // rdx
  const char *v27; // r15
  size_t v28; // r12
  size_t v29; // rax
  char *v30; // rdx
  char *v31; // rdi
  __int64 v32; // rsi
  __int64 v33; // rdx
  __int64 v34; // rax
  void **v35; // r14
  char *v36; // rdi
  size_t v37; // rdx
  void *v38; // [rsp+20h] [rbp-120h]
  size_t v39; // [rsp+38h] [rbp-108h] BYREF
  char *v40; // [rsp+40h] [rbp-100h] BYREF
  size_t n; // [rsp+48h] [rbp-F8h]
  _QWORD src[2]; // [rsp+50h] [rbp-F0h] BYREF
  __int16 v43; // [rsp+60h] [rbp-E0h]
  int v44; // [rsp+70h] [rbp-D0h] BYREF
  __int64 v45; // [rsp+78h] [rbp-C8h]
  int v46; // [rsp+80h] [rbp-C0h]
  __int64 v47; // [rsp+88h] [rbp-B8h]
  void *dest; // [rsp+90h] [rbp-B0h]
  size_t v49; // [rsp+98h] [rbp-A8h]
  _QWORD v50[2]; // [rsp+A0h] [rbp-A0h] BYREF
  _QWORD *v51; // [rsp+B0h] [rbp-90h]
  __int64 v52; // [rsp+B8h] [rbp-88h]
  _QWORD v53[2]; // [rsp+C0h] [rbp-80h] BYREF
  __int64 v54; // [rsp+D0h] [rbp-70h]
  unsigned int v55; // [rsp+D8h] [rbp-68h]
  char v56; // [rsp+DCh] [rbp-64h]
  void *v57; // [rsp+E0h] [rbp-60h] BYREF
  void **v58; // [rsp+E8h] [rbp-58h]
  __int64 v59; // [rsp+100h] [rbp-40h]
  char v60; // [rsp+108h] [rbp-38h]

  dest = v50;
  v44 = 0;
  v45 = 0;
  v47 = 0;
  v49 = 0;
  LOBYTE(v50[0]) = 0;
  v51 = v53;
  v52 = 0;
  LOBYTE(v53[0]) = 0;
  v55 = 1;
  v54 = 0;
  v56 = 0;
  v2 = sub_C33320();
  sub_C3B1B0((__int64)&v40, 0.0);
  sub_C407B0(&v57, (__int64 *)&v40, v2);
  sub_C338F0((__int64)&v40);
  v3 = *(_DWORD *)(a1 + 152);
  v60 = 0;
  v59 = 0;
  if ( v3 != -1 )
  {
    v44 = 1;
    v46 = v3;
    goto LABEL_3;
  }
  v24 = *(_QWORD *)(a1 + 8);
  v44 = 3;
  v25 = sub_BD5D20(v24);
  v40 = (char *)src;
  v27 = v25;
  v28 = v26;
  v29 = v26;
  if ( &v27[v26] && !v27 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v39 = v26;
  if ( v26 > 0xF )
  {
    v40 = (char *)sub_22409D0(&v40, &v39, 0);
    v36 = v40;
    src[0] = v39;
LABEL_64:
    memcpy(v36, v27, v28);
    v29 = v39;
    v30 = v40;
    goto LABEL_50;
  }
  if ( v26 == 1 )
  {
    LOBYTE(src[0]) = *v27;
    v30 = (char *)src;
    goto LABEL_50;
  }
  if ( v26 )
  {
    v36 = (char *)src;
    goto LABEL_64;
  }
  v30 = (char *)src;
LABEL_50:
  n = v29;
  v30[v29] = 0;
  v31 = (char *)dest;
  if ( v40 == (char *)src )
  {
    v37 = n;
    if ( n )
    {
      if ( n == 1 )
        *(_BYTE *)dest = src[0];
      else
        memcpy(dest, src, n);
      v37 = n;
      v31 = (char *)dest;
    }
    v49 = v37;
    v31[v37] = 0;
    v31 = v40;
    goto LABEL_54;
  }
  if ( dest == v50 )
  {
    dest = v40;
    v49 = n;
    v50[0] = src[0];
  }
  else
  {
    v32 = v50[0];
    dest = v40;
    v49 = n;
    v50[0] = src[0];
    if ( v31 )
    {
      v40 = v31;
      src[0] = v32;
      goto LABEL_54;
    }
  }
  v40 = (char *)src;
  v31 = (char *)src;
LABEL_54:
  n = 0;
  *v31 = 0;
  if ( v40 != (char *)src )
    j_j___libc_free_0(v40, src[0] + 1LL);
LABEL_3:
  v4 = *(_QWORD *)a1;
  v5 = sub_1213420(*(_QWORD *)a1 + 1280LL, (__int64)&v44);
  v6 = v4 + 1288;
  v38 = sub_C33340();
  if ( v5 == v4 + 1288 )
  {
    v16 = 0;
  }
  else
  {
    v7 = *(_QWORD *)(v5 + 216);
    if ( v7 == v5 + 200 )
    {
LABEL_25:
      v18 = sub_220F330(v5, v6);
      sub_1209010(*(_QWORD *)(v18 + 208));
      v19 = *(_QWORD *)(v18 + 176);
      if ( v19 )
        j_j___libc_free_0_0(v19);
      if ( *(void **)(v18 + 144) == v38 )
      {
        v33 = *(_QWORD *)(v18 + 152);
        if ( v33 )
        {
          v34 = 24LL * *(_QWORD *)(v33 - 8);
          v35 = (void **)(v33 + v34);
          if ( v33 != v33 + v34 )
          {
            do
            {
              v35 -= 3;
              if ( *v35 == v38 )
                sub_969EE0((__int64)v35);
              else
                sub_C338F0((__int64)v35);
            }
            while ( *(void ***)(v18 + 152) != v35 );
          }
          j_j_j___libc_free_0_0(v35 - 1);
        }
      }
      else
      {
        sub_C338F0(v18 + 144);
      }
      if ( *(_DWORD *)(v18 + 136) > 0x40u )
      {
        v20 = *(_QWORD *)(v18 + 128);
        if ( v20 )
          j_j___libc_free_0_0(v20);
      }
      v21 = *(_QWORD *)(v18 + 96);
      if ( v21 != v18 + 112 )
        j_j___libc_free_0(v21, *(_QWORD *)(v18 + 112) + 1LL);
      v22 = *(_QWORD *)(v18 + 64);
      if ( v22 != v18 + 80 )
        j_j___libc_free_0(v22, *(_QWORD *)(v18 + 80) + 1LL);
      j_j___libc_free_0(v18, 240);
      --*(_QWORD *)(v4 + 1320);
      v16 = 0;
    }
    else
    {
      while ( 1 )
      {
        v13 = *(_QWORD **)(v7 + 192);
        v14 = *(_QWORD *)(v7 + 40);
        if ( *(_DWORD *)(v7 + 32) == 2 )
        {
          v8 = sub_121DBC0((__int64 *)a1, v7 + 64, v14);
          if ( !v8 )
          {
LABEL_11:
            v15 = *(_QWORD *)a1;
            v16 = 1;
            v40 = "referenced value is not a basic block";
            v43 = 259;
            sub_11FD800(v15 + 176, *(_QWORD *)(v7 + 40), (__int64)&v40, 1);
            goto LABEL_12;
          }
        }
        else
        {
          v8 = sub_121E0D0(a1, *(_DWORD *)(v7 + 48), v14);
          if ( !v8 )
            goto LABEL_11;
        }
        v9 = sub_ACC1C0(*(_QWORD *)(a1 + 8), (__int64)v8);
        v10 = v13[1];
        v11 = *(_QWORD *)a1;
        v43 = 260;
        v40 = (char *)(v7 + 64);
        v12 = sub_120A960(v11, *(_QWORD *)(v7 + 40), (__int64)&v40, v10, v9);
        if ( !v12 )
          break;
        sub_BD84D0((__int64)v13, v12);
        sub_B30810(v13);
        v7 = sub_220EEE0(v7);
        if ( v5 + 200 == v7 )
        {
          v4 = *(_QWORD *)a1;
          v6 = *(_QWORD *)a1 + 1288LL;
          goto LABEL_25;
        }
      }
      v16 = 1;
    }
  }
LABEL_12:
  if ( v59 )
    j_j___libc_free_0_0(v59);
  if ( v38 == v57 )
  {
    if ( v58 )
    {
      for ( i = &v58[3 * (_QWORD)*(v58 - 1)]; v58 != i; sub_969EE0((__int64)i) )
      {
        while ( 1 )
        {
          i -= 3;
          if ( v38 == *i )
            break;
          sub_C338F0((__int64)i);
          if ( v58 == i )
            goto LABEL_43;
        }
      }
LABEL_43:
      j_j_j___libc_free_0_0(i - 1);
    }
  }
  else
  {
    sub_C338F0((__int64)&v57);
  }
  if ( v55 > 0x40 && v54 )
    j_j___libc_free_0_0(v54);
  if ( v51 != v53 )
    j_j___libc_free_0(v51, v53[0] + 1LL);
  if ( dest != v50 )
    j_j___libc_free_0(dest, v50[0] + 1LL);
  return v16;
}
