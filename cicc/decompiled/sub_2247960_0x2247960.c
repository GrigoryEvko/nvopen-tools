// Function: sub_2247960
// Address: 0x2247960
//
_QWORD *__fastcall sub_2247960(
        __int64 a1,
        _QWORD *a2,
        __int64 a3,
        __int64 *a4,
        __int64 a5,
        _DWORD *a6,
        __int64 a7,
        __int64 a8,
        __int64 a9,
        _DWORD *a10)
{
  __int64 *v10; // r12
  void *v11; // rsp
  int v13; // edx
  __int64 v14; // r13
  unsigned __int64 v15; // r14
  __int64 v16; // rax
  int v17; // eax
  __int64 v18; // r12
  __int64 v19; // rsi
  size_t v20; // r13
  __int64 v21; // rax
  size_t v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rcx
  unsigned __int64 v25; // rax
  char v26; // r8
  _DWORD *v27; // rax
  int v28; // eax
  char v29; // al
  char v30; // si
  unsigned __int64 v31; // rsi
  __int64 v32; // r8
  int v33; // r12d
  int v34; // eax
  const wchar_t *v35; // r12
  int v36; // r15d
  size_t v37; // r13
  size_t v38; // rbx
  char v39; // al
  char v40; // cl
  int v41; // eax
  wchar_t v42; // r14d
  unsigned __int64 v43; // rax
  char v44; // r14
  _DWORD *v45; // rax
  int v46; // eax
  int *v47; // rax
  int *v48; // rax
  int v49; // edi
  __int64 v50; // rax
  __int64 v51; // rax
  int v52; // eax
  int *v53; // rax
  int v54; // edi
  _DWORD *v55; // rax
  __int64 v56; // rax
  int v57; // eax
  _DWORD *v58; // [rsp+0h] [rbp-70h]
  __int64 v59; // [rsp+8h] [rbp-68h]
  size_t v60; // [rsp+10h] [rbp-60h]
  __int64 v61; // [rsp+18h] [rbp-58h]
  __int64 *v62; // [rsp+20h] [rbp-50h] BYREF
  __int64 v63; // [rsp+28h] [rbp-48h]
  _QWORD *v64; // [rsp+30h] [rbp-40h] BYREF
  __int64 v65; // [rsp+38h] [rbp-38h]

  v64 = a2;
  v62 = a4;
  v65 = a3;
  v63 = a5;
  v58 = a6;
  v10 = (__int64 *)sub_2243120((_QWORD *)(a9 + 208), (__int64)a2);
  v11 = alloca(4 * a8 + 8);
  if ( sub_2247850((__int64)&v64, (__int64)&v62) )
    goto LABEL_2;
  v13 = sub_2247910((__int64)&v64);
  if ( !a8 )
    goto LABEL_2;
  v14 = 0;
  v15 = 0;
  do
  {
    while ( **(_DWORD **)(a7 + 8 * v14) != v13 )
    {
      v16 = *v10;
      LODWORD(v61) = v13;
      v17 = (*(__int64 (__fastcall **)(__int64 *))(v16 + 48))(v10);
      v13 = v61;
      if ( (_DWORD)v61 == v17 )
        break;
      if ( a8 == ++v14 )
        goto LABEL_10;
    }
    *((_DWORD *)&v58 + v15++) = v14++;
  }
  while ( a8 != v14 );
LABEL_10:
  v60 = 0;
  if ( v15 <= 1 )
    goto LABEL_33;
  do
  {
    v18 = (int)v58;
    v19 = 1;
    v20 = wcslen(*(const wchar_t **)(a7 + 8LL * (int)v58));
    do
    {
      v21 = *((int *)&v58 + v19);
      v61 = v19;
      v22 = wcslen(*(const wchar_t **)(a7 + 8 * v21));
      if ( v20 > v22 )
        v20 = v22;
      v19 = v61 + 1;
    }
    while ( v61 + 1 < v15 );
    v25 = v64[2];
    if ( v25 >= v64[3] )
      (*(void (__fastcall **)(_QWORD *))(*v64 + 80LL))(v64);
    else
      v64[2] = v25 + 4;
    ++v60;
    LODWORD(v65) = -1;
    if ( v60 >= v20 )
      goto LABEL_2;
    v26 = 1;
    if ( v64 )
    {
      v27 = (_DWORD *)v64[2];
      v28 = (unsigned __int64)v27 >= v64[3]
          ? (*(__int64 (__fastcall **)(_QWORD *, __int64, __int64, __int64, __int64))(*v64 + 72LL))(
              v64,
              v19,
              v23,
              v24,
              1)
          : *v27;
      v26 = 0;
      if ( v28 == -1 )
      {
        v64 = 0;
        v26 = 1;
      }
    }
    v29 = (_DWORD)v63 == -1;
    v30 = v29 & (v62 != 0);
    if ( v30 )
    {
      v48 = (int *)v62[2];
      if ( (unsigned __int64)v48 >= v62[3] )
      {
        v51 = *v62;
        LOBYTE(v59) = v30;
        LOBYTE(v61) = v26;
        v52 = (*(__int64 (**)(void))(v51 + 72))();
        v30 = v59;
        v26 = v61;
        v49 = v52;
      }
      else
      {
        v49 = *v48;
      }
      v29 = 0;
      if ( v49 == -1 )
      {
        v62 = 0;
        v29 = v30;
      }
    }
    if ( v26 == v29 )
      goto LABEL_2;
    v31 = 0;
    v32 = 4 * v60;
    while ( 1 )
    {
      v33 = *(_DWORD *)(*(_QWORD *)(a7 + 8 * v18) + v32);
      v34 = v65;
      if ( v64 && (_DWORD)v65 == -1 )
      {
        v47 = (int *)v64[2];
        if ( (unsigned __int64)v47 >= v64[3] )
        {
          v50 = *v64;
          v59 = v32;
          v61 = v31;
          v34 = (*(__int64 (**)(void))(v50 + 72))();
          v32 = v59;
          v31 = v61;
        }
        else
        {
          v34 = *v47;
        }
        if ( v34 == -1 )
          v64 = 0;
      }
      if ( v33 == v34 )
        break;
      --v15;
      *((_DWORD *)&v58 + v31) = *((_DWORD *)&v58 + v15);
      if ( v15 <= v31 )
        goto LABEL_32;
LABEL_27:
      v18 = *((int *)&v58 + v31);
    }
    if ( v15 > ++v31 )
      goto LABEL_27;
LABEL_32:
    ;
  }
  while ( v15 > 1 );
LABEL_33:
  if ( v15 == 1 )
  {
    sub_2240940(v64);
    LODWORD(v65) = -1;
    v35 = *(const wchar_t **)(a7 + 8LL * (int)v58);
    v36 = (int)v58;
    v37 = v60 + 1;
    v38 = wcslen(v35);
    if ( v37 < v38 )
    {
      while ( 1 )
      {
        v44 = 1;
        if ( v64 )
        {
          v45 = (_DWORD *)v64[2];
          v46 = (unsigned __int64)v45 >= v64[3] ? (*(__int64 (**)(void))(*v64 + 72LL))() : *v45;
          v44 = 0;
          if ( v46 == -1 )
          {
            v64 = 0;
            v44 = 1;
          }
        }
        v39 = (_DWORD)v63 == -1;
        v40 = v39 & (v62 != 0);
        if ( v40 )
        {
          v53 = (int *)v62[2];
          if ( (unsigned __int64)v53 >= v62[3] )
          {
            v56 = *v62;
            LOBYTE(v61) = v40;
            v57 = (*(__int64 (**)(void))(v56 + 72))();
            v40 = v61;
            v54 = v57;
          }
          else
          {
            v54 = *v53;
          }
          v39 = 0;
          if ( v54 == -1 )
          {
            v62 = 0;
            v39 = v40;
          }
        }
        if ( v44 == v39 )
          break;
        v41 = v65;
        v42 = v35[v37];
        if ( (_DWORD)v65 == -1 && v64 )
        {
          v55 = (_DWORD *)v64[2];
          v41 = (unsigned __int64)v55 >= v64[3] ? (*(__int64 (**)(void))(*v64 + 72LL))() : *v55;
          if ( v41 == -1 )
            v64 = 0;
        }
        if ( v42 != v41 )
          break;
        v43 = v64[2];
        if ( v43 >= v64[3] )
          (*(void (__fastcall **)(_QWORD *))(*v64 + 80LL))(v64);
        else
          v64[2] = v43 + 4;
        ++v37;
        LODWORD(v65) = -1;
        if ( v37 >= v38 )
          goto LABEL_70;
      }
    }
    else
    {
LABEL_70:
      if ( v38 == v37 )
      {
        *v58 = v36;
        return v64;
      }
    }
  }
LABEL_2:
  *a10 |= 4u;
  return v64;
}
