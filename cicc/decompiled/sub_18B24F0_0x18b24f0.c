// Function: sub_18B24F0
// Address: 0x18b24f0
//
__int64 __fastcall sub_18B24F0(_QWORD *a1, char a2)
{
  __int64 v3; // rax
  __int64 v4; // rdi
  _QWORD *v5; // r12
  _QWORD *v6; // rbx
  _BYTE *v7; // r14
  _QWORD *v8; // rax
  _QWORD *v9; // r15
  const char *v10; // rax
  unsigned __int64 v11; // rdx
  _QWORD *i; // rbx
  _QWORD *v13; // rax
  _QWORD *v14; // r12
  _QWORD *v15; // rcx
  __int64 **v16; // rax
  int v17; // esi
  __int64 *v18; // rdi
  __int64 v19; // rax
  __int64 *v20; // rax
  __int64 v21; // rdx
  __int64 *v22; // r14
  __int64 *v23; // r12
  __int64 v24; // r15
  __int64 v25; // rax
  __int64 *v26; // rax
  __int64 v27; // rdx
  const char *v28; // rax
  unsigned __int64 v29; // rdx
  __int64 v30; // rdi
  __int64 v31; // rax
  __int64 v32; // rbx
  __int64 v33; // r14
  __int64 v34; // r12
  __int64 v35; // rdx
  _QWORD *v36; // rax
  unsigned __int64 v37; // rdx
  const char *v39; // rax
  unsigned __int64 v40; // rdx
  _BYTE *v41; // rdx
  _BYTE *v42; // rdx
  _BYTE *v44; // [rsp+8h] [rbp-138h]
  __int64 v45; // [rsp+20h] [rbp-120h] BYREF
  _BYTE *v46; // [rsp+28h] [rbp-118h]
  _BYTE *v47; // [rsp+30h] [rbp-110h]
  __int64 v48; // [rsp+38h] [rbp-108h]
  int v49; // [rsp+40h] [rbp-100h]
  _BYTE v50[72]; // [rsp+48h] [rbp-F8h] BYREF
  __int64 v51; // [rsp+90h] [rbp-B0h] BYREF
  __int64 v52; // [rsp+98h] [rbp-A8h]
  __int64 v53; // [rsp+A0h] [rbp-A0h]
  __int64 v54; // [rsp+A8h] [rbp-98h]
  __int64 v55; // [rsp+B0h] [rbp-90h]
  __int64 v56; // [rsp+B8h] [rbp-88h]
  __int64 v57; // [rsp+C0h] [rbp-80h]
  __int64 v58; // [rsp+C8h] [rbp-78h]
  __int64 v59; // [rsp+D0h] [rbp-70h]
  __int64 v60; // [rsp+D8h] [rbp-68h]
  __int64 v61; // [rsp+E0h] [rbp-60h]
  __int64 v62; // [rsp+E8h] [rbp-58h]
  __int64 v63; // [rsp+F0h] [rbp-50h]
  __int64 v64; // [rsp+F8h] [rbp-48h]
  __int64 v65; // [rsp+100h] [rbp-40h]
  char v66; // [rsp+108h] [rbp-38h]

  v45 = 0;
  v46 = v50;
  v47 = v50;
  v48 = 8;
  v49 = 0;
  v3 = sub_16321C0((__int64)a1, (__int64)"llvm.used", 9, 0);
  if ( v3 )
    sub_18B2370(v3, (__int64)&v45);
  v4 = sub_16321C0((__int64)a1, (__int64)"llvm.compiler.used", 18, 0);
  if ( v4 )
    sub_18B2370(v4, (__int64)&v45);
  v5 = (_QWORD *)a1[2];
  v6 = a1 + 1;
  if ( a1 + 1 != v5 )
  {
    while ( 1 )
    {
      if ( !v5 )
        BUG();
      if ( (*(_BYTE *)(v5 - 3) & 0xFu) - 7 > 1 )
        goto LABEL_11;
      v8 = v46;
      v9 = v5 - 7;
      if ( v47 == v46 )
      {
        v7 = &v46[8 * HIDWORD(v48)];
        if ( v46 == v7 )
        {
          v42 = v46;
        }
        else
        {
          do
          {
            if ( v9 == (_QWORD *)*v8 )
              break;
            ++v8;
          }
          while ( v7 != (_BYTE *)v8 );
          v42 = &v46[8 * HIDWORD(v48)];
        }
LABEL_22:
        while ( v42 != (_BYTE *)v8 )
        {
          if ( *v8 < 0xFFFFFFFFFFFFFFFELL )
            goto LABEL_10;
          ++v8;
        }
        if ( v7 != (_BYTE *)v8 )
          goto LABEL_11;
      }
      else
      {
        v7 = &v47[8 * (unsigned int)v48];
        v8 = sub_16CC9F0((__int64)&v45, (__int64)(v5 - 7));
        if ( v9 == (_QWORD *)*v8 )
        {
          if ( v47 == v46 )
            v42 = &v47[8 * HIDWORD(v48)];
          else
            v42 = &v47[8 * (unsigned int)v48];
          goto LABEL_22;
        }
        if ( v47 == v46 )
        {
          v8 = &v47[8 * HIDWORD(v48)];
          v42 = v8;
          goto LABEL_22;
        }
        v8 = &v47[8 * (unsigned int)v48];
LABEL_10:
        if ( v7 != (_BYTE *)v8 )
          goto LABEL_11;
      }
      if ( a2 && (v10 = sub_1649960((__int64)(v5 - 7)), v11 > 7) && *(_QWORD *)v10 == 0x6762642E6D766C6CLL )
      {
LABEL_11:
        v5 = (_QWORD *)v5[1];
        if ( v6 == v5 )
          break;
      }
      else
      {
        LOWORD(v53) = 257;
        sub_164B780((__int64)(v5 - 7), &v51);
        v5 = (_QWORD *)v5[1];
        if ( v6 == v5 )
          break;
      }
    }
  }
  for ( i = (_QWORD *)a1[4]; a1 + 3 != i; i = (_QWORD *)i[1] )
  {
    if ( !i )
      BUG();
    if ( (*(_BYTE *)(i - 3) & 0xFu) - 7 > 1 )
      goto LABEL_36;
    v13 = v46;
    v14 = i - 7;
    if ( v47 == v46 )
    {
      v15 = &v46[8 * HIDWORD(v48)];
      if ( v46 == (_BYTE *)v15 )
      {
        v41 = v46;
      }
      else
      {
        do
        {
          if ( v14 == (_QWORD *)*v13 )
            break;
          ++v13;
        }
        while ( v15 != v13 );
        v41 = &v46[8 * HIDWORD(v48)];
      }
    }
    else
    {
      v44 = &v47[8 * (unsigned int)v48];
      v13 = sub_16CC9F0((__int64)&v45, (__int64)(i - 7));
      v15 = v44;
      if ( v14 == (_QWORD *)*v13 )
      {
        if ( v47 == v46 )
          v41 = &v47[8 * HIDWORD(v48)];
        else
          v41 = &v47[8 * (unsigned int)v48];
      }
      else
      {
        if ( v47 != v46 )
        {
          v13 = &v47[8 * (unsigned int)v48];
          goto LABEL_35;
        }
        v13 = &v47[8 * HIDWORD(v48)];
        v41 = v13;
      }
    }
    while ( v41 != (_BYTE *)v13 && *v13 >= 0xFFFFFFFFFFFFFFFELL )
      ++v13;
LABEL_35:
    if ( v13 == v15 )
    {
      if ( !a2 || (v39 = sub_1649960((__int64)(i - 7)), v40 <= 7) || *(_QWORD *)v39 != 0x6762642E6D766C6CLL )
      {
        LOWORD(v53) = 257;
        sub_164B780((__int64)(i - 7), &v51);
      }
    }
LABEL_36:
    v16 = (__int64 **)i[6];
    if ( v16 )
    {
      v17 = *((_DWORD *)v16 + 2);
      if ( v17 )
      {
        v18 = *v16;
        v19 = **v16;
        if ( v19 != -8 && v19 )
        {
          v22 = v18;
        }
        else
        {
          v20 = v18 + 1;
          do
          {
            do
            {
              v21 = *v20;
              v22 = v20++;
            }
            while ( v21 == -8 );
          }
          while ( !v21 );
        }
        v23 = &v18[v17];
        while ( v23 != v22 )
        {
          v24 = *(_QWORD *)(*v22 + 8);
          v25 = v22[1];
          if ( v25 != -8 && v25 )
          {
            ++v22;
          }
          else
          {
            v26 = v22 + 2;
            do
            {
              do
              {
                v27 = *v26;
                v22 = v26++;
              }
              while ( v27 == -8 );
            }
            while ( !v27 );
          }
          if ( *(_BYTE *)(v24 + 16) > 3u || (*(_BYTE *)(v24 + 32) & 0xFu) - 7 <= 1 )
          {
            if ( !a2 || (v28 = sub_1649960(v24), v29 <= 7) || *(_QWORD *)v28 != 0x6762642E6D766C6CLL )
            {
              LOWORD(v53) = 257;
              sub_164B780(v24, &v51);
            }
          }
        }
      }
    }
  }
  v51 = 0;
  v52 = 0;
  v53 = 0;
  v54 = 0;
  v55 = 0;
  v56 = 0;
  v57 = 0;
  v58 = 0;
  v59 = 0;
  v60 = 0;
  v61 = 0;
  v62 = 0;
  v63 = 0;
  v64 = 0;
  v65 = 0;
  v66 = 0;
  sub_1648080((__int64)&v51, a1, 0);
  v30 = v63;
  v31 = (v64 - v63) >> 3;
  if ( (_DWORD)v31 )
  {
    v32 = 0;
    v33 = 8LL * (unsigned int)v31;
    do
    {
      v34 = *(_QWORD *)(v30 + v32);
      if ( (*(_BYTE *)(v34 + 9) & 4) == 0 )
      {
        sub_1643640(*(_QWORD *)(v30 + v32));
        if ( !v35 || a2 && (v36 = (_QWORD *)sub_1643640(v34), v37 > 7) && *v36 == 0x6762642E6D766C6CLL )
        {
          v30 = v63;
        }
        else
        {
          sub_1643660((__int64 **)v34, byte_3F871B3, 0);
          v30 = v63;
        }
      }
      v32 += 8;
    }
    while ( v32 != v33 );
  }
  if ( v30 )
    j_j___libc_free_0(v30, v65 - v30);
  j___libc_free_0(v60);
  j___libc_free_0(v56);
  j___libc_free_0(v52);
  if ( v47 != v46 )
    _libc_free((unsigned __int64)v47);
  return 1;
}
