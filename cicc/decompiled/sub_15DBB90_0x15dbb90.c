// Function: sub_15DBB90
// Address: 0x15dbb90
//
void __fastcall sub_15DBB90(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 v3; // r15
  unsigned __int64 v4; // rbx
  __int64 v5; // rbx
  __int64 v6; // r14
  __int64 *v7; // r15
  __int64 *v8; // r12
  __int64 *v9; // rax
  __int64 v10; // rdx
  unsigned __int64 v11; // r9
  __int64 *v12; // rax
  unsigned __int64 v13; // r9
  __int64 v14; // rdx
  unsigned int v15; // eax
  __int64 v16; // rbx
  __int64 *v17; // rdx
  __int64 v18; // r12
  __int64 v19; // r13
  __int64 *v20; // rax
  bool v21; // zf
  unsigned __int64 v22; // r15
  __int64 *v23; // rax
  _QWORD *v24; // rbx
  _QWORD *v25; // r12
  unsigned __int64 v26; // rdi
  _QWORD *v27; // rbx
  _QWORD *v28; // r12
  unsigned __int64 v29; // rdi
  __int64 v30; // rax
  __int64 v31; // rsi
  unsigned __int64 v32; // rdx
  char v33; // r8
  _QWORD *v34; // rax
  unsigned __int64 v35; // rdi
  char v36; // al
  _QWORD *v37; // r15
  unsigned __int64 v38; // rdi
  unsigned __int64 v39; // [rsp-120h] [rbp-120h]
  unsigned __int64 v40; // [rsp-120h] [rbp-120h]
  __int64 *v41; // [rsp-118h] [rbp-118h]
  __int64 *v42; // [rsp-118h] [rbp-118h]
  __int64 v43; // [rsp-108h] [rbp-108h]
  __int64 v44; // [rsp-F8h] [rbp-F8h]
  unsigned int v45; // [rsp-F0h] [rbp-F0h]
  unsigned __int64 v46; // [rsp-F0h] [rbp-F0h]
  unsigned __int64 v47; // [rsp-E8h] [rbp-E8h] BYREF
  unsigned __int64 v48; // [rsp-E0h] [rbp-E0h] BYREF
  __int64 *v49; // [rsp-D8h] [rbp-D8h] BYREF
  __int64 v50; // [rsp-D0h] [rbp-D0h]
  _BYTE v51[64]; // [rsp-C8h] [rbp-C8h] BYREF
  __int64 v52; // [rsp-88h] [rbp-88h] BYREF
  _QWORD *v53; // [rsp-80h] [rbp-80h]
  __int64 v54; // [rsp-78h] [rbp-78h]
  unsigned int v55; // [rsp-70h] [rbp-70h]
  __int64 v56; // [rsp-68h] [rbp-68h] BYREF
  _QWORD *v57; // [rsp-60h] [rbp-60h]
  __int64 v58; // [rsp-58h] [rbp-58h]
  unsigned int v59; // [rsp-50h] [rbp-50h]
  char v60; // [rsp-48h] [rbp-48h]

  if ( !a3 )
    return;
  v3 = a1;
  if ( a3 == 1 )
  {
    v30 = a2[1];
    v31 = *a2;
    v32 = v30 & 0xFFFFFFFFFFFFFFF8LL;
    if ( (v30 & 4) != 0 )
      sub_15D4360(a1, v31, v32);
    else
      sub_15D7E60(a1, v31, v32);
    return;
  }
  v52 = 0;
  v49 = (__int64 *)v51;
  v50 = 0x400000000LL;
  v53 = 0;
  v54 = 0;
  v55 = 0;
  v56 = 0;
  v57 = 0;
  v58 = 0;
  v59 = 0;
  v60 = 0;
  sub_15DB440(a2, a3, (__int64)&v49);
  v45 = v50;
  v44 = (unsigned int)v50;
  if ( (_DWORD)v50 )
  {
    ++v52;
    v4 = (((((((4 * (int)v50 / 3u + 1) | ((unsigned __int64)(4 * (int)v50 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v50 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v50 / 3u + 1) >> 1)) >> 4)
         | (((4 * (int)v50 / 3u + 1) | ((unsigned __int64)(4 * (int)v50 / 3u + 1) >> 1)) >> 2)
         | (4 * (int)v50 / 3u + 1)
         | ((unsigned __int64)(4 * (int)v50 / 3u + 1) >> 1)) >> 8)
       | (((((4 * (int)v50 / 3u + 1) | ((unsigned __int64)(4 * (int)v50 / 3u + 1) >> 1)) >> 2)
         | (4 * (int)v50 / 3u + 1)
         | ((unsigned __int64)(4 * (int)v50 / 3u + 1) >> 1)) >> 4)
       | (((4 * (int)v50 / 3u + 1) | ((unsigned __int64)(4 * (int)v50 / 3u + 1) >> 1)) >> 2)
       | (4 * (int)v50 / 3u + 1)
       | ((unsigned __int64)(4 * (int)v50 / 3u + 1) >> 1);
    v5 = ((v4 >> 16) | v4) + 1;
    if ( v55 < (unsigned int)v5 )
      sub_15CE860((__int64)&v52, v5);
    ++v56;
    if ( v59 < (unsigned int)v5 )
      sub_15CE860((__int64)&v56, v5);
    v6 = 2LL * (unsigned int)v50;
    if ( v49 != &v49[v6] )
    {
      v7 = v49;
      v8 = &v49[v6];
      do
      {
        v48 = *v7;
        v9 = sub_15CFBA0((__int64)&v52, (__int64 *)&v48);
        v10 = *((unsigned int *)v9 + 4);
        v11 = v7[1] & 0xFFFFFFFFFFFFFFFCLL;
        if ( (unsigned int)v10 >= *((_DWORD *)v9 + 5) )
        {
          v40 = v7[1] & 0xFFFFFFFFFFFFFFFCLL;
          v42 = v9;
          sub_16CD150(v9 + 1, v9 + 3, 0, 8);
          v9 = v42;
          v11 = v40;
          v10 = *((unsigned int *)v42 + 4);
        }
        *(_QWORD *)(v9[1] + 8 * v10) = v11;
        ++*((_DWORD *)v9 + 4);
        v48 = v7[1] & 0xFFFFFFFFFFFFFFF8LL;
        v12 = sub_15CFBA0((__int64)&v56, (__int64 *)&v48);
        v13 = v7[1] & 4 | *v7 & 0xFFFFFFFFFFFFFFFBLL;
        v14 = *((unsigned int *)v12 + 4);
        if ( (unsigned int)v14 >= *((_DWORD *)v12 + 5) )
        {
          v39 = v7[1] & 4 | *v7 & 0xFFFFFFFFFFFFFFFBLL;
          v41 = v12;
          sub_16CD150(v12 + 1, v12 + 3, 0, 8);
          v12 = v41;
          v13 = v39;
          v14 = *((unsigned int *)v41 + 4);
        }
        v7 += 2;
        *(_QWORD *)(v12[1] + 8 * v14) = v13;
        ++*((_DWORD *)v12 + 4);
      }
      while ( v8 != v7 );
      v3 = a1;
    }
  }
  else
  {
    ++v52;
    ++v56;
  }
  v15 = *(_DWORD *)(v3 + 40);
  if ( v15 > 0x64 )
  {
    if ( *(_DWORD *)(v3 + 40) / 0x28u >= v45 )
      goto LABEL_18;
LABEL_55:
    sub_15D3360(v3, (__int64)&v49);
    goto LABEL_18;
  }
  if ( v45 > v15 )
    goto LABEL_55;
LABEL_18:
  v16 = 0;
  if ( v44 )
  {
    v43 = v3;
    while ( !v60 )
    {
      v17 = &v49[2 * (unsigned int)v50 - 2];
      v18 = *v17;
      v19 = v17[1];
      LODWORD(v50) = v50 - 1;
      v48 = v18;
      v20 = sub_15CFBA0((__int64)&v52, (__int64 *)&v48);
      v21 = (*((_DWORD *)v20 + 4))-- == 1;
      if ( v21 )
      {
        v47 = v18;
        v36 = sub_15CFAE0((__int64)&v52, (__int64 *)&v47, &v48);
        v37 = (_QWORD *)v48;
        if ( v36 )
        {
          v38 = *(_QWORD *)(v48 + 8);
          if ( v38 != v48 + 24 )
            _libc_free(v38);
          *v37 = -16;
          LODWORD(v54) = v54 - 1;
          ++HIDWORD(v54);
        }
      }
      v22 = v19 & 0xFFFFFFFFFFFFFFF8LL;
      v48 = v19 & 0xFFFFFFFFFFFFFFF8LL;
      v23 = sub_15CFBA0((__int64)&v56, (__int64 *)&v48);
      v21 = (*((_DWORD *)v23 + 4))-- == 1;
      if ( v21 )
      {
        v47 = v19 & 0xFFFFFFFFFFFFFFF8LL;
        v33 = sub_15CFAE0((__int64)&v56, (__int64 *)&v47, &v48);
        v34 = (_QWORD *)v48;
        if ( v33 )
        {
          v35 = *(_QWORD *)(v48 + 8);
          if ( v35 != v48 + 24 )
          {
            v46 = v48;
            _libc_free(v35);
            v34 = (_QWORD *)v46;
          }
          *v34 = -16;
          LODWORD(v58) = v58 - 1;
          ++HIDWORD(v58);
        }
      }
      if ( (v19 & 4) != 0 )
      {
        sub_15D4190(v43, (__int64)&v49, v18, v22);
        if ( ++v16 == v44 )
          break;
      }
      else
      {
        sub_15D7CB0(v43, (__int64)&v49, v18, v22);
        if ( ++v16 == v44 )
          break;
      }
    }
  }
  if ( v59 )
  {
    v24 = v57;
    v25 = &v57[7 * v59];
    do
    {
      if ( *v24 != -8 && *v24 != -16 )
      {
        v26 = v24[1];
        if ( (_QWORD *)v26 != v24 + 3 )
          _libc_free(v26);
      }
      v24 += 7;
    }
    while ( v25 != v24 );
  }
  j___libc_free_0(v57);
  if ( v55 )
  {
    v27 = v53;
    v28 = &v53[7 * v55];
    do
    {
      if ( *v27 != -16 && *v27 != -8 )
      {
        v29 = v27[1];
        if ( (_QWORD *)v29 != v27 + 3 )
          _libc_free(v29);
      }
      v27 += 7;
    }
    while ( v28 != v27 );
  }
  j___libc_free_0(v53);
  if ( v49 != (__int64 *)v51 )
    _libc_free((unsigned __int64)v49);
}
