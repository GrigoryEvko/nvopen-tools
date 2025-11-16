// Function: sub_1881E40
// Address: 0x1881e40
//
__int64 __fastcall sub_1881E40(__int64 a1, _BYTE *a2, unsigned __int64 a3, _QWORD *a4)
{
  _BYTE *v6; // rsi
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // rdx
  __int64 v9; // rsi
  unsigned __int64 v10; // rdi
  __int64 v11; // rdx
  __int64 (__fastcall *v12)(__int64, __int64 *); // rax
  __int64 result; // rax
  _BYTE *v14; // rdi
  _QWORD *v15; // rdi
  _QWORD *v16; // r13
  char *v17; // r10
  char *v18; // r8
  _QWORD *v19; // r15
  signed __int64 v20; // r9
  char *v21; // rcx
  char *v22; // rax
  _BYTE *v23; // rdx
  _QWORD *v24; // rax
  signed __int64 v25; // rdx
  __int64 v26; // rax
  __int64 v27; // rdx
  _BYTE *v28; // rsi
  char **v29; // r10
  _BYTE *v30; // rax
  unsigned __int64 v31; // r8
  __int64 v32; // rax
  _QWORD *v33; // rbx
  size_t v34; // rdx
  _QWORD *v35; // rax
  _QWORD *v36; // rdx
  _QWORD *v37; // r9
  signed __int64 v38; // r8
  _BOOL8 v39; // rdi
  _BYTE *v40; // rsi
  _QWORD *v41; // rax
  signed __int64 v42; // rdi
  unsigned __int64 v43; // [rsp+8h] [rbp-D8h]
  char **v44; // [rsp+10h] [rbp-D0h]
  unsigned __int64 v45; // [rsp+10h] [rbp-D0h]
  char **v46; // [rsp+18h] [rbp-C8h]
  char *v47; // [rsp+18h] [rbp-C8h]
  char *v48; // [rsp+18h] [rbp-C8h]
  _QWORD *v49; // [rsp+20h] [rbp-C0h]
  __int64 v50; // [rsp+28h] [rbp-B8h]
  _QWORD *v53; // [rsp+38h] [rbp-A8h]
  _QWORD *v54; // [rsp+38h] [rbp-A8h]
  char v55; // [rsp+47h] [rbp-99h] BYREF
  __int64 v56; // [rsp+48h] [rbp-98h] BYREF
  void *src; // [rsp+50h] [rbp-90h] BYREF
  _BYTE *v58; // [rsp+58h] [rbp-88h]
  _BYTE *v59; // [rsp+60h] [rbp-80h]
  const char *v60; // [rsp+70h] [rbp-70h]
  unsigned __int64 v61; // [rsp+78h] [rbp-68h]
  _BYTE *v62; // [rsp+80h] [rbp-60h] BYREF
  unsigned __int64 v63; // [rsp+88h] [rbp-58h]
  __int64 v64[2]; // [rsp+90h] [rbp-50h] BYREF
  _BYTE v65[64]; // [rsp+A0h] [rbp-40h] BYREF

  v60 = byte_3F871B3;
  src = 0;
  v58 = 0;
  v59 = 0;
  v61 = 0;
  v62 = a2;
  v63 = a3;
  if ( a3 )
  {
    while ( 1 )
    {
      LOBYTE(v64[0]) = 44;
      v7 = sub_16D20C0((__int64 *)&v62, (char *)v64, 1u, 0);
      if ( v7 == -1 )
      {
        v9 = (__int64)v62;
        v7 = v63;
        v10 = 0;
        v11 = 0;
      }
      else
      {
        v8 = v7 + 1;
        v9 = (__int64)v62;
        if ( v7 + 1 > v63 )
          v8 = v63;
        v10 = v63 - v8;
        v11 = (__int64)&v62[v8];
        if ( v7 && v7 > v63 )
          v7 = v63;
      }
      v60 = (const char *)v9;
      v63 = v10;
      v62 = (_BYTE *)v11;
      v61 = v7;
      if ( sub_16D2B80(v9, v7, 0, (unsigned __int64 *)v64) )
        break;
      v6 = v58;
      v56 = v64[0];
      if ( v58 == v59 )
      {
        sub_9CA200((__int64)&src, v58, &v56);
        if ( !v63 )
          goto LABEL_19;
      }
      else
      {
        if ( v58 )
        {
          *(_QWORD *)v58 = v64[0];
          v6 = v58;
        }
        v58 = v6 + 8;
        if ( !v63 )
          goto LABEL_19;
      }
    }
    v12 = *(__int64 (__fastcall **)(__int64, __int64 *))(*(_QWORD *)a1 + 232LL);
    v64[0] = (__int64)"key not an integer";
    v65[1] = 1;
    v65[0] = 3;
    result = v12(a1, v64);
    v14 = src;
    if ( !src )
      return result;
    return j_j___libc_free_0(v14, v59 - v14);
  }
LABEL_19:
  v15 = (_QWORD *)a4[2];
  v16 = a4 + 1;
  if ( !v15 )
  {
    v19 = a4 + 1;
    goto LABEL_38;
  }
  v17 = v58;
  v18 = (char *)src;
  v19 = a4 + 1;
  v20 = v58 - (_BYTE *)src;
  do
  {
    v21 = (char *)v15[5];
    v22 = (char *)v15[4];
    if ( v21 - v22 > v20 )
      v21 = &v22[v20];
    v23 = src;
    if ( v22 != v21 )
    {
      while ( *(_QWORD *)v22 >= *(_QWORD *)v23 )
      {
        if ( *(_QWORD *)v22 > *(_QWORD *)v23 )
          goto LABEL_54;
        v22 += 8;
        v23 += 8;
        if ( v21 == v22 )
          goto LABEL_53;
      }
LABEL_28:
      v15 = (_QWORD *)v15[3];
      continue;
    }
LABEL_53:
    if ( v23 != v58 )
      goto LABEL_28;
LABEL_54:
    v19 = v15;
    v15 = (_QWORD *)v15[2];
  }
  while ( v15 );
  if ( v19 == v16 )
    goto LABEL_38;
  v24 = (_QWORD *)v19[4];
  v25 = v19[5] - (_QWORD)v24;
  if ( v20 > v25 )
    v17 = (char *)src + v25;
  if ( src != v17 )
  {
    while ( *(_QWORD *)v18 >= *v24 )
    {
      if ( *(_QWORD *)v18 > *v24 )
        goto LABEL_47;
      v18 += 8;
      ++v24;
      if ( v17 == v18 )
        goto LABEL_58;
    }
LABEL_38:
    v49 = v19;
    v26 = sub_22077B0(80);
    v28 = src;
    v19 = (_QWORD *)v26;
    v29 = (char **)(v26 + 32);
    v30 = v58;
    v19[4] = 0;
    v19[5] = 0;
    v19[6] = 0;
    v31 = v30 - v28;
    if ( v30 == v28 )
    {
      v50 = 0;
      v34 = 0;
      v33 = 0;
    }
    else
    {
      v50 = v30 - v28;
      if ( v31 > 0x7FFFFFFFFFFFFFF8LL )
        sub_4261EA(80, v28, v27);
      v46 = v29;
      v32 = sub_22077B0(v31);
      v28 = src;
      v29 = v46;
      v33 = (_QWORD *)v32;
      v30 = v58;
      v31 = v58 - (_BYTE *)src;
      v34 = v58 - (_BYTE *)src;
    }
    v19[4] = v33;
    v19[5] = v33;
    v19[6] = (char *)v33 + v50;
    if ( v28 == v30 )
    {
      v19[5] = (char *)v33 + v34;
      *((_DWORD *)v19 + 14) = 0;
      v19[8] = 0;
      v19[9] = 0;
      v45 = v31;
      v48 = (char *)v33 + v34;
      v35 = sub_14F7820(a4, v49, v29);
      v37 = v48;
      v38 = v45;
      if ( v36 )
        goto LABEL_43;
      if ( !v33 )
      {
LABEL_62:
        v53 = v35;
        j_j___libc_free_0(v19, 80);
        v19 = v53;
        goto LABEL_47;
      }
    }
    else
    {
      v43 = v31;
      v47 = (char *)v33 + v34;
      v44 = v29;
      memmove(v33, v28, v34);
      *((_DWORD *)v19 + 14) = 0;
      v19[8] = 0;
      v19[5] = v47;
      v19[9] = 0;
      v35 = sub_14F7820(a4, v49, v44);
      v37 = v47;
      v38 = v43;
      if ( v36 )
      {
LABEL_43:
        if ( v16 != v36 && !v35 )
        {
          v41 = (_QWORD *)v36[4];
          v42 = v36[5] - (_QWORD)v41;
          if ( v42 < v38 )
            v37 = (_QWORD *)((char *)v33 + v42);
          if ( v33 == v37 )
          {
LABEL_72:
            v39 = v36[5] != (_QWORD)v41;
            goto LABEL_46;
          }
          while ( *v33 >= *v41 )
          {
            if ( *v33 > *v41 )
            {
              v39 = 0;
              goto LABEL_46;
            }
            ++v33;
            ++v41;
            if ( v37 == v33 )
              goto LABEL_72;
          }
        }
        v39 = 1;
LABEL_46:
        sub_220F040(v39, v19, v36, v16);
        ++a4[5];
        goto LABEL_47;
      }
    }
    v54 = v35;
    j_j___libc_free_0(v33, v50);
    v35 = v54;
    goto LABEL_62;
  }
LABEL_58:
  if ( (_QWORD *)v19[5] != v24 )
    goto LABEL_38;
LABEL_47:
  if ( a2 )
  {
    v64[0] = (__int64)v65;
    sub_18736F0(v64, a2, (__int64)&a2[a3]);
    v40 = (_BYTE *)v64[0];
  }
  else
  {
    v40 = v65;
    v64[1] = 0;
    v64[0] = (__int64)v65;
    v65[0] = 0;
  }
  if ( (*(unsigned __int8 (__fastcall **)(__int64, _BYTE *, __int64, _QWORD, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
         a1,
         v40,
         1,
         0,
         &v55,
         &v56) )
  {
    sub_187A0E0(a1, (__int64)(v19 + 7));
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v56);
  }
  result = sub_2240A30(v64);
  v14 = src;
  if ( src )
    return j_j___libc_free_0(v14, v59 - v14);
  return result;
}
