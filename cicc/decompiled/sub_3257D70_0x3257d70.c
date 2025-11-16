// Function: sub_3257D70
// Address: 0x3257d70
//
void __fastcall sub_3257D70(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rbx
  const char *v10; // rax
  __int64 v11; // rdx
  __int64 v12; // r14
  __int64 v13; // r12
  __int64 v14; // rax
  _QWORD *v15; // rdi
  unsigned int v16; // eax
  __int64 v17; // rdx
  __int64 v18; // rdx
  unsigned __int8 *v19; // r14
  unsigned __int8 *v20; // r8
  int v21; // eax
  unsigned __int64 v22; // rax
  __int64 v23; // rcx
  __int64 v24; // rax
  const char *v25; // rax
  unsigned __int64 v26; // rdx
  __int64 v27; // rax
  __int64 v28; // rax
  unsigned __int64 v29; // rdx
  __int64 v30; // rax
  __int64 v31; // rbx
  __int64 *v32; // r13
  __int64 v33; // rsi
  __int64 *v34; // r13
  __int64 *i; // r14
  __int64 v36; // rsi
  __int64 *v37; // r13
  __int64 *j; // r14
  __int64 v39; // rsi
  unsigned __int64 v40; // rdi
  __int64 v41; // rax
  __int64 v42; // rax
  __int64 *v43; // rsi
  signed __int64 v44; // rdx
  __int64 v45; // rax
  __int64 v46; // rcx
  bool v47; // cf
  unsigned __int64 v48; // rax
  unsigned __int64 v49; // r12
  __int64 v50; // rax
  char *v51; // r15
  unsigned __int64 v52; // r12
  unsigned __int8 *v53; // [rsp+0h] [rbp-B0h]
  __int64 *src; // [rsp+8h] [rbp-A8h]
  char *v55; // [rsp+10h] [rbp-A0h]
  __int64 v57; // [rsp+20h] [rbp-90h]
  __int64 *n; // [rsp+28h] [rbp-88h]
  __int64 *v59; // [rsp+30h] [rbp-80h] BYREF
  __int64 *v60; // [rsp+38h] [rbp-78h]
  __int64 *v61; // [rsp+40h] [rbp-70h]
  _QWORD *v62; // [rsp+50h] [rbp-60h] BYREF
  __int64 v63; // [rsp+58h] [rbp-58h]
  _QWORD v64[10]; // [rsp+60h] [rbp-50h] BYREF

  v6 = (_QWORD *)a1[1];
  v59 = 0;
  v7 = v6[30];
  v60 = 0;
  v61 = 0;
  v8 = *(_QWORD *)(v7 + 2488);
  v9 = *(_QWORD *)(v8 + 32);
  v57 = v8 + 24;
  if ( v9 == v8 + 24 )
  {
    src = 0;
    if ( a1[2] == a1[3] )
      return;
LABEL_83:
    v31 = v6[28];
    (*(void (__fastcall **)(__int64, _QWORD, _QWORD))(*(_QWORD *)v31 + 176LL))(
      v31,
      *(_QWORD *)(*(_QWORD *)(v6[27] + 168LL) + 728LL),
      0);
    goto LABEL_45;
  }
  v55 = 0;
  n = 0;
  src = 0;
  while ( 2 )
  {
    if ( v9 )
    {
      v13 = v9 - 56;
      v14 = v9 - 56;
    }
    else
    {
      v13 = 0;
      v14 = 0;
    }
    v62 = v64;
    v64[0] = v14;
    v15 = v64;
    v63 = 0x400000001LL;
    v16 = 1;
    while ( 1 )
    {
      v17 = v16--;
      v18 = v15[v17 - 1];
      LODWORD(v63) = v16;
      v19 = *(unsigned __int8 **)(v18 + 16);
      if ( v19 )
        break;
LABEL_30:
      if ( !v16 )
      {
        if ( v15 != v64 )
          _libc_free((unsigned __int64)v15);
        goto LABEL_14;
      }
    }
    while ( 1 )
    {
      while ( 1 )
      {
        while ( 1 )
        {
          v20 = (unsigned __int8 *)*((_QWORD *)v19 + 3);
          v21 = *v20;
          if ( (_BYTE)v21 == 4 )
            goto LABEL_28;
          if ( (unsigned __int8)v21 <= 0x1Cu )
            break;
          v22 = (unsigned int)(v21 - 34);
          if ( (unsigned __int8)v22 > 0x33u )
            goto LABEL_7;
          v23 = 0x8000000000041LL;
          if ( !_bittest64(&v23, v22) )
            goto LABEL_7;
          if ( v19 == v20 - 32 && v13 == *(_QWORD *)v19 )
          {
            v19 = (unsigned __int8 *)*((_QWORD *)v19 + 1);
            if ( !v19 )
              goto LABEL_29;
          }
          else
          {
            v24 = sub_B43CB0(*((_QWORD *)v19 + 3));
            v25 = sub_BD5D20(v24);
            if ( v26 <= 0xA )
              goto LABEL_7;
            v27 = (__int64)&v25[v26 - 11];
            if ( *(_QWORD *)v27 != 0x68745F7469786524LL || *(_WORD *)(v27 + 8) != 28277 || *(_BYTE *)(v27 + 10) != 107 )
              goto LABEL_7;
LABEL_28:
            v19 = (unsigned __int8 *)*((_QWORD *)v19 + 1);
            if ( !v19 )
              goto LABEL_29;
          }
        }
        if ( (unsigned __int8)v21 <= 3u )
          break;
        if ( (unsigned __int8)v21 > 0x15u )
          goto LABEL_28;
        v28 = (unsigned int)v63;
        v29 = (unsigned int)v63 + 1LL;
        if ( v29 > HIDWORD(v63) )
        {
          v53 = (unsigned __int8 *)*((_QWORD *)v19 + 3);
          sub_C8D5F0((__int64)&v62, v64, v29, 8u, (__int64)v20, a6);
          v28 = (unsigned int)v63;
          v20 = v53;
        }
        v62[v28] = v20;
        LODWORD(v63) = v63 + 1;
        v19 = (unsigned __int8 *)*((_QWORD *)v19 + 1);
        if ( !v19 )
        {
LABEL_29:
          v16 = v63;
          v15 = v62;
          goto LABEL_30;
        }
      }
      v10 = sub_BD5D20(*((_QWORD *)v19 + 3));
      if ( v11 != 22
        || *(_QWORD *)v10 ^ 0x6D72612E6D766C6CLL | *((_QWORD *)v10 + 1) ^ 0x6D79732E63653436LL
        || *((_DWORD *)v10 + 4) != 1835822946
        || *((_WORD *)v10 + 10) != 28769 )
      {
        break;
      }
      v19 = (unsigned __int8 *)*((_QWORD *)v19 + 1);
      if ( !v19 )
        goto LABEL_29;
    }
LABEL_7:
    if ( v62 != v64 )
      _libc_free((unsigned __int64)v62);
    if ( (*(_BYTE *)(v13 + 33) & 3) == 1 )
    {
      v41 = sub_31DB510(a1[1], v13);
      v42 = sub_3257CF0((__int64)a1, v41);
      v62 = (_QWORD *)v42;
      if ( v42 )
      {
        v43 = v60;
        if ( v60 == v61 )
        {
          sub_E8AD60((__int64)&v59, v60, &v62);
        }
        else
        {
          if ( v60 )
          {
            *v60 = v42;
            v43 = v60;
          }
          v60 = v43 + 1;
        }
      }
    }
    v12 = sub_31DB510(a1[1], v13);
    if ( v55 == (char *)n )
    {
      v44 = v55 - (char *)src;
      v45 = (v55 - (char *)src) >> 3;
      if ( v45 == 0xFFFFFFFFFFFFFFFLL )
        sub_4262D8((__int64)"vector::_M_realloc_insert");
      v46 = 1;
      if ( v45 )
        v46 = (v55 - (char *)src) >> 3;
      v47 = __CFADD__(v46, v45);
      v48 = v46 + v45;
      if ( v47 )
      {
        v49 = 0x7FFFFFFFFFFFFFF8LL;
      }
      else
      {
        if ( !v48 )
        {
          v52 = 0;
          v51 = 0;
LABEL_68:
          if ( &v51[v44] )
            *(_QWORD *)&v51[v44] = v12;
          n = (__int64 *)&v51[v44 + 8];
          if ( v44 > 0 )
          {
            memmove(v51, src, v44);
          }
          else if ( !src )
          {
LABEL_72:
            v55 = (char *)v52;
            src = (__int64 *)v51;
            goto LABEL_14;
          }
          j_j___libc_free_0((unsigned __int64)src);
          goto LABEL_72;
        }
        if ( v48 > 0xFFFFFFFFFFFFFFFLL )
          v48 = 0xFFFFFFFFFFFFFFFLL;
        v49 = 8 * v48;
      }
      v50 = sub_22077B0(v49);
      v44 = v55 - (char *)src;
      v51 = (char *)v50;
      v52 = v50 + v49;
      goto LABEL_68;
    }
    if ( n )
      *n = v12;
    ++n;
LABEL_14:
    v9 = *(_QWORD *)(v9 + 8);
    if ( v9 != v57 )
      continue;
    break;
  }
  if ( n == src )
  {
    v40 = (unsigned __int64)v59;
    if ( v59 == v60 && a1[3] == a1[2] )
      goto LABEL_50;
    v6 = (_QWORD *)a1[1];
    goto LABEL_83;
  }
  v30 = a1[1];
  v31 = *(_QWORD *)(v30 + 224);
  (*(void (__fastcall **)(__int64, _QWORD, _QWORD))(*(_QWORD *)v31 + 176LL))(
    v31,
    *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v30 + 216) + 168LL) + 728LL),
    0);
  v32 = src;
  do
  {
    v33 = *v32++;
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v31 + 352LL))(v31, v33);
  }
  while ( n != v32 );
LABEL_45:
  (*(void (__fastcall **)(__int64, _QWORD, _QWORD))(*(_QWORD *)v31 + 176LL))(
    v31,
    *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1[1] + 216) + 168LL) + 736LL),
    0);
  v34 = v60;
  for ( i = v59; v34 != i; ++i )
  {
    v36 = *i;
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v31 + 352LL))(v31, v36);
  }
  (*(void (__fastcall **)(__int64, _QWORD, _QWORD))(*(_QWORD *)v31 + 176LL))(
    v31,
    *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1[1] + 216) + 168LL) + 744LL),
    0);
  v37 = (__int64 *)a1[3];
  for ( j = (__int64 *)a1[2]; v37 != j; ++j )
  {
    v39 = *j;
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v31 + 352LL))(v31, v39);
  }
  v40 = (unsigned __int64)v59;
LABEL_50:
  if ( v40 )
    j_j___libc_free_0(v40);
  if ( src )
    j_j___libc_free_0((unsigned __int64)src);
}
