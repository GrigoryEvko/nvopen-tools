// Function: sub_262B910
// Address: 0x262b910
//
void __fastcall sub_262B910(__int64 a1, _BYTE *a2, unsigned __int64 a3, _QWORD *a4)
{
  _BYTE *v6; // rsi
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // rdx
  __int64 v9; // rsi
  unsigned __int64 v10; // rdi
  __int64 v11; // rdx
  void (__fastcall *v12)(__int64, __int64 *); // rax
  void *v13; // rdi
  _QWORD *v14; // rdi
  char *v15; // r10
  char *v16; // r8
  unsigned __int64 v17; // r15
  signed __int64 v18; // r9
  char *v19; // rsi
  char *v20; // rax
  _BYTE *v21; // rdx
  _QWORD *v22; // rax
  signed __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rdx
  _BYTE *v26; // rsi
  _QWORD *v27; // rcx
  char **v28; // r10
  _BYTE *v29; // rax
  unsigned __int64 v30; // r8
  __int64 v31; // r13
  __int64 v32; // rax
  _QWORD *v33; // rbx
  size_t v34; // rdx
  _QWORD *v35; // rax
  _QWORD *v36; // rdx
  _QWORD *v37; // r9
  _QWORD *v38; // rcx
  signed __int64 v39; // r8
  char v40; // di
  _QWORD *v41; // rsi
  _QWORD *v42; // rax
  signed __int64 v43; // rdi
  unsigned __int64 v44; // [rsp+8h] [rbp-E8h]
  char **v45; // [rsp+10h] [rbp-E0h]
  unsigned __int64 v46; // [rsp+10h] [rbp-E0h]
  char **v47; // [rsp+18h] [rbp-D8h]
  _QWORD *v48; // [rsp+18h] [rbp-D8h]
  _QWORD *v49; // [rsp+18h] [rbp-D8h]
  char *v50; // [rsp+20h] [rbp-D0h]
  char *v51; // [rsp+20h] [rbp-D0h]
  _QWORD *v52; // [rsp+28h] [rbp-C8h]
  _QWORD *v55; // [rsp+38h] [rbp-B8h]
  _QWORD *v56; // [rsp+38h] [rbp-B8h]
  char v57; // [rsp+47h] [rbp-A9h] BYREF
  __int64 v58; // [rsp+48h] [rbp-A8h] BYREF
  void *src; // [rsp+50h] [rbp-A0h] BYREF
  _BYTE *v60; // [rsp+58h] [rbp-98h]
  _BYTE *v61; // [rsp+60h] [rbp-90h]
  const char *v62; // [rsp+70h] [rbp-80h]
  unsigned __int64 v63; // [rsp+78h] [rbp-78h]
  _BYTE *v64; // [rsp+80h] [rbp-70h] BYREF
  unsigned __int64 v65; // [rsp+88h] [rbp-68h]
  __int64 v66[2]; // [rsp+90h] [rbp-60h] BYREF
  _BYTE v67[16]; // [rsp+A0h] [rbp-50h] BYREF
  char v68; // [rsp+B0h] [rbp-40h]
  char v69; // [rsp+B1h] [rbp-3Fh]

  src = 0;
  v60 = 0;
  v61 = 0;
  v62 = byte_3F871B3;
  v63 = 0;
  v64 = a2;
  v65 = a3;
  if ( a3 )
  {
    while ( 1 )
    {
      LOBYTE(v66[0]) = 44;
      v7 = sub_C931B0((__int64 *)&v64, v66, 1u, 0);
      if ( v7 == -1 )
      {
        v9 = (__int64)v64;
        v7 = v65;
        v10 = 0;
        v11 = 0;
      }
      else
      {
        v8 = v7 + 1;
        v9 = (__int64)v64;
        if ( v7 + 1 > v65 )
        {
          v8 = v65;
          v10 = 0;
        }
        else
        {
          v10 = v65 - v8;
        }
        v11 = (__int64)&v64[v8];
        if ( v7 > v65 )
          v7 = v65;
      }
      v62 = (const char *)v9;
      v65 = v10;
      v64 = (_BYTE *)v11;
      v63 = v7;
      if ( sub_C93C90(v9, v7, 0, (unsigned __int64 *)v66) )
        break;
      v6 = v60;
      v58 = v66[0];
      if ( v60 == v61 )
      {
        sub_9CA200((__int64)&src, v60, &v58);
        if ( !v65 )
          goto LABEL_19;
      }
      else
      {
        if ( v60 )
        {
          *(_QWORD *)v60 = v66[0];
          v6 = v60;
        }
        v60 = v6 + 8;
        if ( !v65 )
          goto LABEL_19;
      }
    }
    v12 = *(void (__fastcall **)(__int64, __int64 *))(*(_QWORD *)a1 + 248LL);
    v69 = 1;
    v66[0] = (__int64)"key not an integer";
    v68 = 3;
    v12(a1, v66);
    v13 = src;
    if ( !src )
      return;
LABEL_14:
    j_j___libc_free_0((unsigned __int64)v13);
    return;
  }
LABEL_19:
  v14 = (_QWORD *)a4[2];
  if ( !v14 )
  {
    v17 = (unsigned __int64)(a4 + 1);
    goto LABEL_38;
  }
  v15 = v60;
  v16 = (char *)src;
  v17 = (unsigned __int64)(a4 + 1);
  v18 = v60 - (_BYTE *)src;
  do
  {
    v19 = (char *)v14[5];
    v20 = (char *)v14[4];
    if ( v19 - v20 > v18 )
      v19 = &v20[v18];
    v21 = src;
    if ( v20 != v19 )
    {
      while ( *(_QWORD *)v20 >= *(_QWORD *)v21 )
      {
        if ( *(_QWORD *)v20 > *(_QWORD *)v21 )
          goto LABEL_56;
        v20 += 8;
        v21 += 8;
        if ( v19 == v20 )
          goto LABEL_55;
      }
LABEL_28:
      v14 = (_QWORD *)v14[3];
      continue;
    }
LABEL_55:
    if ( v60 != v21 )
      goto LABEL_28;
LABEL_56:
    v17 = (unsigned __int64)v14;
    v14 = (_QWORD *)v14[2];
  }
  while ( v14 );
  if ( (_QWORD *)v17 == a4 + 1 )
    goto LABEL_38;
  v22 = *(_QWORD **)(v17 + 32);
  v23 = *(_QWORD *)(v17 + 40) - (_QWORD)v22;
  if ( v18 > v23 )
    v15 = (char *)src + v23;
  if ( src != v15 )
  {
    while ( *(_QWORD *)v16 >= *v22 )
    {
      if ( *(_QWORD *)v16 > *v22 )
        goto LABEL_47;
      v16 += 8;
      ++v22;
      if ( v15 == v16 )
        goto LABEL_60;
    }
LABEL_38:
    v52 = (_QWORD *)v17;
    v24 = sub_22077B0(0x50u);
    v26 = src;
    v27 = a4 + 1;
    v17 = v24;
    v28 = (char **)(v24 + 32);
    v29 = v60;
    *(_QWORD *)(v17 + 32) = 0;
    *(_QWORD *)(v17 + 40) = 0;
    v30 = v29 - v26;
    *(_QWORD *)(v17 + 48) = 0;
    if ( v29 == v26 )
    {
      v34 = 0;
      v31 = 0;
      v33 = 0;
    }
    else
    {
      v31 = v29 - v26;
      if ( v30 > 0x7FFFFFFFFFFFFFF8LL )
        sub_4261EA(80, v26, v25);
      v47 = v28;
      v32 = sub_22077B0(v30);
      v26 = src;
      v27 = a4 + 1;
      v33 = (_QWORD *)v32;
      v29 = v60;
      v28 = v47;
      v30 = v60 - (_BYTE *)src;
      v34 = v60 - (_BYTE *)src;
    }
    *(_QWORD *)(v17 + 32) = v33;
    *(_QWORD *)(v17 + 40) = v33;
    *(_QWORD *)(v17 + 48) = (char *)v33 + v31;
    if ( v29 == v26 )
    {
      *(_QWORD *)(v17 + 40) = (char *)v33 + v34;
      *(_DWORD *)(v17 + 56) = 0;
      *(_QWORD *)(v17 + 64) = 0;
      *(_QWORD *)(v17 + 72) = 0;
      v46 = v30;
      v49 = v27;
      v51 = (char *)v33 + v34;
      v35 = sub_9D7C50(a4, v52, v28);
      v37 = v51;
      v38 = v49;
      v39 = v46;
      if ( v36 )
        goto LABEL_43;
      if ( !v33 )
      {
LABEL_64:
        v55 = v35;
        j_j___libc_free_0(v17);
        v17 = (unsigned __int64)v55;
        goto LABEL_47;
      }
    }
    else
    {
      v44 = v30;
      v48 = v27;
      v50 = (char *)v33 + v34;
      v45 = v28;
      memmove(v33, v26, v34);
      *(_DWORD *)(v17 + 56) = 0;
      *(_QWORD *)(v17 + 64) = 0;
      *(_QWORD *)(v17 + 40) = v50;
      *(_QWORD *)(v17 + 72) = 0;
      v35 = sub_9D7C50(a4, v52, v45);
      v37 = v50;
      v38 = v48;
      v39 = v44;
      if ( v36 )
      {
LABEL_43:
        if ( v38 != v36 && !v35 )
        {
          v42 = (_QWORD *)v36[4];
          v43 = v36[5] - (_QWORD)v42;
          if ( v43 < v39 )
            v37 = (_QWORD *)((char *)v33 + v43);
          if ( v33 == v37 )
          {
LABEL_74:
            v40 = v36[5] != (_QWORD)v42;
            goto LABEL_46;
          }
          while ( *v33 >= *v42 )
          {
            if ( *v33 > *v42 )
            {
              v40 = 0;
              goto LABEL_46;
            }
            ++v33;
            ++v42;
            if ( v37 == v33 )
              goto LABEL_74;
          }
        }
        v40 = 1;
LABEL_46:
        sub_220F040(v40, v17, v36, v38);
        ++a4[5];
        goto LABEL_47;
      }
    }
    v56 = v35;
    j_j___libc_free_0((unsigned __int64)v33);
    v35 = v56;
    goto LABEL_64;
  }
LABEL_60:
  if ( *(_QWORD **)(v17 + 40) != v22 )
    goto LABEL_38;
LABEL_47:
  if ( a2 )
  {
    v66[0] = (__int64)v67;
    sub_2619AF0(v66, a2, (__int64)&a2[a3]);
    v41 = (_QWORD *)v66[0];
  }
  else
  {
    v66[1] = 0;
    v66[0] = (__int64)v67;
    v41 = v67;
    v67[0] = 0;
  }
  if ( (*(unsigned __int8 (__fastcall **)(__int64, _QWORD *, __int64, _QWORD, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
         a1,
         v41,
         1,
         0,
         &v57,
         &v58) )
  {
    sub_261D790(a1, v17 + 56);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v58);
  }
  if ( (_BYTE *)v66[0] != v67 )
    j_j___libc_free_0(v66[0]);
  v13 = src;
  if ( src )
    goto LABEL_14;
}
