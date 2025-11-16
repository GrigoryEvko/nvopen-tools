// Function: sub_25CF4F0
// Address: 0x25cf4f0
//
void __fastcall sub_25CF4F0(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // r13
  size_t v3; // r12
  int v4; // eax
  __int64 v5; // r15
  _QWORD *v6; // r8
  unsigned __int64 v7; // r15
  unsigned __int64 v8; // rax
  const char **v9; // r14
  size_t v10; // rdx
  __int64 v11; // rax
  _QWORD *v12; // r8
  _QWORD *v13; // rcx
  const char *v14; // rax
  __int64 v15; // rax
  char *v16; // rsi
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 v19; // rdx
  char v20; // al
  __int64 v21; // rsi
  __int64 v22; // r12
  const void *v23; // r14
  size_t v24; // r13
  int v25; // eax
  int v26; // eax
  unsigned __int64 v27; // rax
  unsigned __int64 v28; // rax
  __int64 v29; // rdx
  const void *v30; // r15
  size_t v31; // r14
  int v32; // eax
  unsigned int v33; // r8d
  __int64 *v34; // rcx
  __int64 v35; // r13
  __int64 v36; // r12
  const void *v37; // r13
  size_t v38; // r15
  int v39; // eax
  int v40; // eax
  unsigned __int64 v41; // rax
  unsigned __int64 v42; // r8
  __int64 v43; // r12
  __int64 v44; // rbx
  _QWORD *v45; // rdi
  __int64 v46; // rax
  unsigned int v47; // r8d
  __int64 *v48; // rcx
  __int64 v49; // r13
  __int64 *v50; // rax
  __int64 *v51; // rax
  __int64 v52; // rax
  __int64 v53; // [rsp+8h] [rbp-198h]
  _QWORD *v54; // [rsp+10h] [rbp-190h]
  __int64 *v55; // [rsp+28h] [rbp-178h]
  _QWORD *v56; // [rsp+30h] [rbp-170h]
  __int64 v57; // [rsp+30h] [rbp-170h]
  __int64 *v58; // [rsp+30h] [rbp-170h]
  __int64 v59; // [rsp+38h] [rbp-168h]
  __int64 v60; // [rsp+38h] [rbp-168h]
  unsigned int v61; // [rsp+38h] [rbp-168h]
  _QWORD *v62; // [rsp+40h] [rbp-160h]
  __int64 v63; // [rsp+58h] [rbp-148h] BYREF
  unsigned __int64 v64; // [rsp+60h] [rbp-140h] BYREF
  __int64 v65; // [rsp+68h] [rbp-138h]
  __int64 v66; // [rsp+70h] [rbp-130h]
  _QWORD *v67; // [rsp+80h] [rbp-120h] BYREF
  char v68; // [rsp+90h] [rbp-110h]
  __int64 v69[2]; // [rsp+A0h] [rbp-100h] BYREF
  __int64 v70; // [rsp+D0h] [rbp-D0h] BYREF
  int v71; // [rsp+D8h] [rbp-C8h] BYREF
  _QWORD *v72; // [rsp+E0h] [rbp-C0h]
  int *v73; // [rsp+E8h] [rbp-B8h]
  int *v74; // [rsp+F0h] [rbp-B0h]
  __int64 v75; // [rsp+F8h] [rbp-A8h]
  unsigned __int16 v76[20]; // [rsp+100h] [rbp-A0h] BYREF
  char v77; // [rsp+128h] [rbp-78h]
  const char **v78; // [rsp+130h] [rbp-70h] BYREF
  size_t v79; // [rsp+138h] [rbp-68h]
  const char *v80; // [rsp+140h] [rbp-60h]
  __int64 v81; // [rsp+148h] [rbp-58h]
  unsigned __int64 v82; // [rsp+150h] [rbp-50h]
  __int64 v83; // [rsp+158h] [rbp-48h]
  __int64 v84; // [rsp+160h] [rbp-40h]

  v1 = *(_QWORD *)(a1 + 24);
  v66 = 0x1000000000LL;
  v2 = *(_QWORD *)(v1 + 24);
  v64 = 0;
  v65 = 0;
  v59 = v1 + 8;
  if ( v2 != v1 + 8 )
  {
    while ( 1 )
    {
      v7 = (v2 + 32) & 0xFFFFFFFFFFFFFFF8LL | *(unsigned __int8 *)(v1 + 343);
      v8 = (v2 + 32) & 0xFFFFFFFFFFFFFFF8LL | *(_BYTE *)(v1 + 343) & 0xF8;
      v9 = *(const char ***)(v8 + 8);
      if ( (v7 & 1) != 0 )
      {
        v9 = (const char **)sub_BD5D20(*(_QWORD *)(v8 + 8));
        v3 = v10;
      }
      else
      {
        v3 = *(_QWORD *)(v8 + 16);
      }
      v80 = (const char *)v7;
      v78 = v9;
      v79 = v3;
      v4 = sub_C92610();
      v5 = (unsigned int)sub_C92740((__int64)&v64, v9, v3, v4);
      v6 = (_QWORD *)(v64 + 8 * v5);
      if ( *v6 )
      {
        if ( *v6 != -8 )
          goto LABEL_6;
        LODWORD(v66) = v66 - 1;
      }
      v62 = (_QWORD *)(v64 + 8 * v5);
      v11 = sub_C7D670(v3 + 17, 8);
      v12 = v62;
      v13 = (_QWORD *)v11;
      if ( v3 )
      {
        v56 = (_QWORD *)v11;
        memcpy((void *)(v11 + 16), v9, v3);
        v12 = v62;
        v13 = v56;
      }
      v14 = v80;
      *((_BYTE *)v13 + v3 + 16) = 0;
      *v13 = v3;
      v13[1] = v14;
      *v12 = v13;
      ++HIDWORD(v65);
      sub_C929D0((__int64 *)&v64, v5);
LABEL_6:
      v2 = sub_220EF30(v2);
      if ( v2 == v59 )
        break;
      v1 = *(_QWORD *)(a1 + 24);
    }
  }
  LOWORD(v82) = 260;
  v78 = (const char **)&qword_4FF0748;
  sub_C7EAD0((__int64)&v67, &v78, 0, 1u, 0);
  if ( (v68 & 1) != 0 && (_DWORD)v67 )
    sub_C64ED0("Failed to open context file", 1u);
  v72 = 0;
  v73 = &v71;
  v74 = &v71;
  v78 = (const char **)byte_3F871B3;
  v80 = byte_3F871B3;
  v75 = 0;
  v79 = 0;
  v81 = 0;
  v82 = 0;
  v83 = 0;
  v84 = 0;
  v15 = v67[2];
  v54 = v67;
  v71 = 0;
  v16 = (char *)v67[1];
  v67 = 0;
  sub_C6EB10((__int64)v76, v16, v15 - (_QWORD)v16);
  v19 = v77 & 1;
  v20 = (2 * v19) | v77 & 0xFD;
  v77 = v20;
  if ( (_BYTE)v19 )
  {
    v69[0] = 0;
    v77 = v20 & 0xFD;
    v52 = *(_QWORD *)v76;
    *(_QWORD *)v76 = 0;
    v63 = v52 | 1;
    sub_9C8CB0(v69);
    sub_C641D0(&v63, 1u);
  }
  v69[1] = (__int64)&v78;
  v21 = (__int64)&v70;
  v69[0] = 0;
  if ( !(unsigned __int8)sub_25CEE40((__int64)v76, &v70, v19, (unsigned int)(2 * v19), v17, v18, 0) )
    sub_C64ED0("Invalid thinlto contextual profile format.", 1u);
  v22 = (__int64)v73;
  v55 = (__int64 *)(a1 + 40);
  if ( v73 != &v71 )
  {
    while ( 1 )
    {
      v23 = *(const void **)(v22 + 32);
      v24 = *(_QWORD *)(v22 + 40);
      v25 = sub_C92610();
      v21 = (__int64)v23;
      v26 = sub_C92860((__int64 *)&v64, v23, v24, v25);
      if ( v26 != -1 )
      {
        v27 = v64 + 8LL * v26;
        if ( v27 != v64 + 8LL * (unsigned int)v65 )
        {
          v28 = *(_QWORD *)(*(_QWORD *)v27 + 8LL) & 0xFFFFFFFFFFFFFFF8LL;
          v29 = *(_QWORD *)(v28 + 24);
          if ( *(_QWORD *)(v28 + 32) - v29 == 8 )
            break;
        }
      }
LABEL_20:
      v22 = sub_220EEE0(v22);
      if ( (int *)v22 == &v71 )
        goto LABEL_33;
    }
    v30 = *(const void **)(*(_QWORD *)v29 + 24LL);
    v31 = *(_QWORD *)(*(_QWORD *)v29 + 32LL);
    v32 = sub_C92610();
    v21 = a1;
    v33 = sub_C92740((__int64)v55, v30, v31, v32);
    v34 = (__int64 *)(*(_QWORD *)(a1 + 40) + 8LL * v33);
    v35 = *v34;
    if ( *v34 )
    {
      if ( v35 != -8 )
      {
LABEL_26:
        v60 = *(_QWORD *)(v22 + 72);
        if ( v60 != *(_QWORD *)(v22 + 64) )
        {
          v57 = v35;
          v53 = v22;
          v36 = *(_QWORD *)(v22 + 64);
          do
          {
            v37 = *(const void **)v36;
            v38 = *(_QWORD *)(v36 + 8);
            v39 = sub_C92610();
            v21 = (__int64)v37;
            v40 = sub_C92860((__int64 *)&v64, v37, v38, v39);
            if ( v40 != -1 )
            {
              v41 = v64 + 8LL * v40;
              if ( v41 != v64 + 8LL * (unsigned int)v65 )
              {
                v21 = v57 + 8;
                sub_25CF280((__int64)v69, v57 + 8, (_QWORD *)(*(_QWORD *)v41 + 8LL));
              }
            }
            v36 += 32;
          }
          while ( v60 != v36 );
          v22 = v53;
        }
        goto LABEL_20;
      }
      --*(_DWORD *)(a1 + 56);
    }
    v58 = v34;
    v61 = v33;
    v46 = sub_C7D670(v31 + 41, 8);
    v47 = v61;
    v48 = v58;
    v49 = v46;
    if ( v31 )
    {
      memcpy((void *)(v46 + 40), v30, v31);
      v47 = v61;
      v48 = v58;
    }
    *(_BYTE *)(v49 + v31 + 40) = 0;
    v21 = v47;
    *(_QWORD *)v49 = v31;
    *(_QWORD *)(v49 + 8) = 0;
    *(_QWORD *)(v49 + 16) = 0;
    *(_QWORD *)(v49 + 24) = 0;
    *(_DWORD *)(v49 + 32) = 0;
    *v48 = v49;
    ++*(_DWORD *)(a1 + 52);
    v50 = (__int64 *)(*(_QWORD *)(a1 + 40) + 8LL * (unsigned int)sub_C929D0(v55, v47));
    v35 = *v50;
    if ( *v50 == -8 || !v35 )
    {
      v51 = v50 + 1;
      do
      {
        do
          v35 = *v51++;
        while ( v35 == -8 );
      }
      while ( !v35 );
    }
    goto LABEL_26;
  }
LABEL_33:
  if ( (v77 & 2) != 0 )
    sub_125BF00(v76, v21);
  if ( (v77 & 1) != 0 )
  {
    if ( *(_QWORD *)v76 )
      (*(void (__fastcall **)(_QWORD))(**(_QWORD **)v76 + 8LL))(*(_QWORD *)v76);
  }
  else
  {
    sub_C6BC50(v76);
  }
  if ( v82 )
    j_j___libc_free_0(v82);
  sub_25CD350(v72);
  (*(void (__fastcall **)(_QWORD *))(*v54 + 8LL))(v54);
  if ( (v68 & 1) == 0 && v67 )
    (*(void (__fastcall **)(_QWORD *))(*v67 + 8LL))(v67);
  v42 = v64;
  if ( HIDWORD(v65) && (_DWORD)v65 )
  {
    v43 = 8LL * (unsigned int)v65;
    v44 = 0;
    do
    {
      v45 = *(_QWORD **)(v42 + v44);
      if ( v45 != (_QWORD *)-8LL && v45 )
      {
        sub_C7D6A0((__int64)v45, *v45 + 17LL, 8);
        v42 = v64;
      }
      v44 += 8;
    }
    while ( v44 != v43 );
  }
  _libc_free(v42);
}
