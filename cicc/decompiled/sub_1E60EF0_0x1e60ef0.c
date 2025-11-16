// Function: sub_1E60EF0
// Address: 0x1e60ef0
//
_QWORD *__fastcall sub_1E60EF0(_QWORD *a1, __int64 a2, __int64 a3)
{
  _QWORD *v3; // r15
  char *v5; // rax
  __int64 v6; // r8
  int v7; // r9d
  __int64 v8; // rax
  __int64 v9; // rcx
  int v10; // ebx
  unsigned int v11; // r14d
  __int64 v12; // r15
  __int64 v13; // rdx
  int v14; // eax
  unsigned int v15; // eax
  __int64 v16; // rax
  __int64 v17; // rbx
  unsigned int v18; // eax
  __int64 v19; // rcx
  int v20; // r8d
  int v21; // r9d
  __int64 v22; // r13
  unsigned int v23; // r12d
  __int64 v24; // r15
  __int64 v25; // r10
  __int64 *v26; // rax
  __int64 v27; // rdx
  __int64 v28; // r13
  int v29; // r8d
  __int64 v30; // rcx
  unsigned int v31; // eax
  __int64 *v32; // r12
  __int64 v33; // rsi
  unsigned __int64 v34; // rdi
  unsigned int v35; // ebx
  __int64 v36; // rcx
  __int64 v37; // r8
  unsigned int v38; // r9d
  __int64 v39; // rax
  unsigned int v40; // r11d
  __int64 *v41; // r13
  int v42; // r14d
  _QWORD *v43; // r10
  __int64 v44; // r14
  _QWORD *v45; // rsi
  __int64 v46; // rdx
  __int64 *v47; // rax
  unsigned int v48; // r11d
  _QWORD *v49; // rbx
  _QWORD *v50; // r12
  unsigned __int64 v51; // rdi
  _QWORD *v52; // rbx
  _QWORD *v53; // r12
  unsigned __int64 v54; // rdi
  __int64 *v56; // rdi
  __int64 *v57; // rsi
  __int64 v59; // [rsp+20h] [rbp-140h]
  int v60; // [rsp+28h] [rbp-138h]
  int v61; // [rsp+30h] [rbp-130h]
  __int64 v62; // [rsp+30h] [rbp-130h]
  __int64 v63; // [rsp+38h] [rbp-128h]
  __int64 v64; // [rsp+40h] [rbp-120h]
  __int64 v65; // [rsp+58h] [rbp-108h] BYREF
  char *v66; // [rsp+60h] [rbp-100h] BYREF
  _QWORD *v67; // [rsp+68h] [rbp-F8h]
  char *v68; // [rsp+70h] [rbp-F0h]
  __int64 v69; // [rsp+78h] [rbp-E8h] BYREF
  _QWORD *v70; // [rsp+80h] [rbp-E0h]
  __int64 v71; // [rsp+88h] [rbp-D8h]
  unsigned int v72; // [rsp+90h] [rbp-D0h]
  __int64 v73; // [rsp+98h] [rbp-C8h]
  __int64 v74[4]; // [rsp+A0h] [rbp-C0h] BYREF
  _QWORD *v75; // [rsp+C0h] [rbp-A0h]
  unsigned int v76; // [rsp+D0h] [rbp-90h]
  unsigned __int64 *v77; // [rsp+E0h] [rbp-80h] BYREF
  __int64 *v78; // [rsp+E8h] [rbp-78h]
  __int64 *v79; // [rsp+F0h] [rbp-70h] BYREF
  __int64 v80; // [rsp+F8h] [rbp-68h]
  int v81; // [rsp+100h] [rbp-60h]
  _BYTE v82[88]; // [rsp+108h] [rbp-58h] BYREF

  v3 = a1;
  *a1 = a1 + 2;
  a1[1] = 0x400000000LL;
  v5 = (char *)sub_22077B0(8);
  v59 = a3;
  *(_QWORD *)v5 = 0;
  v66 = v5;
  v68 = v5 + 8;
  v67 = v5 + 8;
  v69 = 0;
  v70 = 0;
  v71 = 0;
  v72 = 0;
  v73 = a3;
  sub_1E5FE90((__int64)&v66);
  v8 = *(_QWORD *)(a2 + 88);
  v9 = v8 + 320;
  v64 = v8 + 320;
  if ( *(_QWORD *)(v8 + 328) == v8 + 320 )
    goto LABEL_47;
  v10 = 0;
  v11 = 1;
  v12 = *(_QWORD *)(v8 + 328);
  do
  {
    while ( 1 )
    {
      v61 = v10;
      v74[0] = v12;
      ++v10;
      sub_1E5F7F0((__int64)&v77, v12, v59, v9, v6, v7);
      v14 = (int)v78;
      if ( v77 != (unsigned __int64 *)&v79 )
      {
        v60 = (int)v78;
        _libc_free((unsigned __int64)v77);
        v14 = v60;
      }
      if ( !v14 )
        break;
      v12 = *(_QWORD *)(v12 + 8);
      if ( v64 == v12 )
        goto LABEL_8;
    }
    sub_1E05890((__int64)a1, v74, v13, v9, v6, v7);
    v15 = sub_1E60810((__int64)&v66, v74[0], v11, (unsigned __int8 (__fastcall *)(__int64))sub_1E5E510, 1);
    v12 = *(_QWORD *)(v12 + 8);
    v11 = v15;
  }
  while ( v64 != v12 );
LABEL_8:
  v3 = a1;
  if ( v11 == v61 + 2 )
    goto LABEL_47;
  v77 = 0;
  v78 = (__int64 *)v82;
  v79 = (__int64 *)v82;
  v80 = 4;
  v16 = *(_QWORD *)(a2 + 88);
  v81 = 0;
  v17 = *(_QWORD *)(v16 + 328);
  v62 = v16 + 320;
  if ( v16 + 320 == v17 )
    goto LABEL_26;
  do
  {
    v65 = v17;
    if ( (unsigned __int8)sub_1E5F140((__int64)&v69, &v65, v74) )
      goto LABEL_23;
    v18 = sub_1E60B80((__int64)&v66, v17, v11, (unsigned __int8 (__fastcall *)(__int64))sub_1E5E510, v11);
    v22 = v18;
    v23 = v18;
    v24 = 8LL * v18;
    v25 = *(_QWORD *)&v66[v24];
    v26 = v78;
    v74[0] = v25;
    if ( v79 != v78 )
      goto LABEL_12;
    v27 = HIDWORD(v80);
    v56 = &v78[HIDWORD(v80)];
    if ( v78 != v56 )
    {
      v57 = 0;
      while ( 1 )
      {
        v27 = *v26;
        if ( v25 == *v26 )
          goto LABEL_13;
        if ( v27 == -2 )
          v57 = v26;
        if ( v56 == ++v26 )
        {
          if ( !v57 )
            break;
          *v57 = v25;
          --v81;
          v77 = (unsigned __int64 *)((char *)v77 + 1);
          goto LABEL_13;
        }
      }
    }
    if ( HIDWORD(v80) < (unsigned int)v80 )
    {
      ++HIDWORD(v80);
      *v56 = v25;
      v77 = (unsigned __int64 *)((char *)v77 + 1);
    }
    else
    {
LABEL_12:
      sub_16CCBA0((__int64)&v77, v25);
    }
LABEL_13:
    sub_1E05890((__int64)a1, v74, v27, v19, v20, v21);
    if ( v23 > v11 )
    {
      v28 = 8 * (v22 - (v23 + ~v11));
      while ( 1 )
      {
        if ( v72 )
        {
          v29 = 1;
          v30 = *(_QWORD *)&v66[v24];
          v31 = (v72 - 1) & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
          v32 = &v70[9 * v31];
          v33 = *v32;
          if ( v30 == *v32 )
          {
LABEL_18:
            v34 = v32[5];
            if ( (__int64 *)v34 != v32 + 7 )
              _libc_free(v34);
            *v32 = -16;
            LODWORD(v71) = v71 - 1;
            ++HIDWORD(v71);
          }
          else
          {
            while ( v33 != -8 )
            {
              v31 = (v72 - 1) & (v29 + v31);
              v32 = &v70[9 * v31];
              v33 = *v32;
              if ( v30 == *v32 )
                goto LABEL_18;
              ++v29;
            }
          }
        }
        --v67;
        if ( v28 == v24 )
          break;
        v24 -= 8;
      }
    }
    v11 = sub_1E60810((__int64)&v66, v74[0], v11, (unsigned __int8 (__fastcall *)(__int64))sub_1E5E510, 1);
LABEL_23:
    v17 = *(_QWORD *)(v17 + 8);
  }
  while ( v62 != v17 );
  v3 = a1;
  if ( v79 != v78 )
    _libc_free((unsigned __int64)v79);
LABEL_26:
  v35 = 0;
  sub_1E5F0D0((__int64)v74, v59);
  v39 = 0;
  if ( *((_DWORD *)v3 + 2) )
  {
    do
    {
      while ( 1 )
      {
        v41 = (__int64 *)(*v3 + 8 * v39);
        sub_1E5F7F0((__int64)&v77, *v41, v59, v36, v37, v38);
        v42 = (int)v78;
        if ( v77 != (unsigned __int64 *)&v79 )
          _libc_free((unsigned __int64)v77);
        if ( !v42 )
          break;
        sub_1E5F230((__int64)v74);
        v38 = sub_1E60B80((__int64)v74, *v41, 0, (unsigned __int8 (__fastcall *)(__int64))sub_1E5E510, 0);
        if ( v38 <= 1 )
          break;
        v43 = (_QWORD *)*v3;
        LODWORD(v37) = 2;
        v44 = v74[0];
        v63 = *((unsigned int *)v3 + 2);
        v45 = (_QWORD *)(*v3 + v63 * 8);
        while ( 1 )
        {
          v77 = *(unsigned __int64 **)(v44 + 8LL * (unsigned int)v37);
          if ( v45 != sub_1E5E530(v43, (__int64)v45, (__int64 *)&v77) )
            break;
          v37 = (unsigned int)(v37 + 1);
          if ( v38 < (unsigned int)v37 )
            goto LABEL_29;
        }
        v46 = *v41;
        v47 = &v43[v63 - 1];
        v36 = *v47;
        *v41 = *v47;
        *v47 = v46;
        v48 = *((_DWORD *)v3 + 2) - 1;
        v39 = v35;
        *((_DWORD *)v3 + 2) = v48;
        if ( v48 <= v35 )
          goto LABEL_38;
      }
      v40 = *((_DWORD *)v3 + 2);
LABEL_29:
      v39 = ++v35;
    }
    while ( v40 > v35 );
  }
LABEL_38:
  if ( v76 )
  {
    v49 = v75;
    v50 = &v75[9 * v76];
    do
    {
      if ( *v49 != -16 && *v49 != -8 )
      {
        v51 = v49[5];
        if ( (_QWORD *)v51 != v49 + 7 )
          _libc_free(v51);
      }
      v49 += 9;
    }
    while ( v50 != v49 );
  }
  j___libc_free_0(v75);
  if ( v74[0] )
    j_j___libc_free_0(v74[0], v74[2] - v74[0]);
LABEL_47:
  if ( v72 )
  {
    v52 = v70;
    v53 = &v70[9 * v72];
    do
    {
      if ( *v52 != -16 && *v52 != -8 )
      {
        v54 = v52[5];
        if ( (_QWORD *)v54 != v52 + 7 )
          _libc_free(v54);
      }
      v52 += 9;
    }
    while ( v53 != v52 );
  }
  j___libc_free_0(v70);
  if ( v66 )
    j_j___libc_free_0(v66, v68 - v66);
  return v3;
}
