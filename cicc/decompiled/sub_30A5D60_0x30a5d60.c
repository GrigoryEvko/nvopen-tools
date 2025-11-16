// Function: sub_30A5D60
// Address: 0x30a5d60
//
__int64 __fastcall sub_30A5D60(__int64 a1)
{
  __int64 *v1; // rdx
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v4; // rax
  __int64 v5; // r12
  int v6; // r13d
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rdi
  __int64 (*v10)(); // rax
  __int64 v11; // rsi
  void *v12; // rdi
  char *v13; // r13
  char *v14; // r12
  __int64 v15; // r15
  char *v16; // rbx
  void *v17; // rax
  void *v18; // r14
  __int64 v19; // rax
  __int64 *v20; // r12
  char *v21; // rbx
  int v22; // r13d
  __int64 v23; // rax
  unsigned __int64 v24; // r14
  const char *v25; // rax
  size_t v26; // rdx
  _QWORD *v27; // rax
  bool v28; // al
  char *v29; // rcx
  char *v30; // rdx
  void *v31; // rax
  char *v32; // rbx
  __int64 *v33; // rbx
  __int64 v34; // rax
  _QWORD *v35; // rax
  __int64 v36; // r13
  unsigned __int64 v37; // r8
  __int64 v38; // r14
  __int64 v39; // r13
  _QWORD *v40; // rdi
  unsigned int v41; // eax
  char *v42; // r14
  int v43; // r12d
  int v44; // ebx
  unsigned int v45; // r13d
  __int64 v46; // rdi
  __int64 v47; // rax
  __int64 v48; // rdi
  __int64 (*v49)(); // rax
  __int64 **v51; // [rsp+8h] [rbp-128h]
  __int64 v52; // [rsp+10h] [rbp-120h]
  unsigned int v54; // [rsp+20h] [rbp-110h]
  unsigned __int8 v55; // [rsp+24h] [rbp-10Ch]
  unsigned __int8 v56; // [rsp+25h] [rbp-10Bh]
  char v57; // [rsp+26h] [rbp-10Ah]
  char v58; // [rsp+27h] [rbp-109h]
  __int64 v59; // [rsp+28h] [rbp-108h]
  __int64 v60; // [rsp+30h] [rbp-100h]
  char *v61; // [rsp+38h] [rbp-F8h]
  char v62; // [rsp+38h] [rbp-F8h]
  unsigned int v63; // [rsp+38h] [rbp-F8h]
  _QWORD *v64; // [rsp+40h] [rbp-F0h] BYREF
  __int64 v65; // [rsp+48h] [rbp-E8h]
  _QWORD v66[2]; // [rsp+50h] [rbp-E0h] BYREF
  _QWORD v67[2]; // [rsp+60h] [rbp-D0h] BYREF
  void *dest; // [rsp+70h] [rbp-C0h]
  void *v69; // [rsp+78h] [rbp-B8h]
  char *v70; // [rsp+80h] [rbp-B0h]
  _QWORD v71[2]; // [rsp+90h] [rbp-A0h] BYREF
  __int64 v72; // [rsp+A0h] [rbp-90h]
  __int64 v73; // [rsp+A8h] [rbp-88h]
  __int64 v74; // [rsp+B0h] [rbp-80h]
  unsigned __int64 v75; // [rsp+B8h] [rbp-78h]
  __int64 v76; // [rsp+C0h] [rbp-70h]
  __int64 v77; // [rsp+C8h] [rbp-68h]
  void *src; // [rsp+D0h] [rbp-60h]
  char *v79; // [rsp+D8h] [rbp-58h]
  __int64 v80; // [rsp+E0h] [rbp-50h]
  unsigned __int64 v81; // [rsp+E8h] [rbp-48h]
  __int64 v82; // [rsp+F0h] [rbp-40h]
  __int64 v83; // [rsp+F8h] [rbp-38h]

  v1 = *(__int64 **)(a1 + 8);
  v2 = *v1;
  v3 = v1[1];
  if ( v2 == v3 )
LABEL_97:
    BUG();
  while ( *(_UNKNOWN **)v2 != &unk_4F86A88 )
  {
    v2 += 16;
    if ( v3 == v2 )
      goto LABEL_97;
  }
  v51 = *(__int64 ***)((*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v2 + 8) + 104LL))(
                         *(_QWORD *)(v2 + 8),
                         &unk_4F86A88)
                     + 176);
  v4 = *(unsigned int *)(a1 + 200);
  if ( (_DWORD)v4 )
  {
    v5 = 0;
    v6 = 0;
    v7 = 8 * v4;
    while ( 1 )
    {
      while ( 1 )
      {
        v8 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(*(_QWORD *)(a1 + 192) + v5) + 120LL))(*(_QWORD *)(*(_QWORD *)(a1 + 192) + v5));
        if ( !v8 )
          break;
        v6 |= (*(__int64 (__fastcall **)(__int64, __int64 *))(*(_QWORD *)(v8 - 176) + 24LL))(v8 - 176, *v51);
LABEL_8:
        v5 += 8;
        if ( v7 == v5 )
          goto LABEL_12;
      }
      v9 = *(_QWORD *)(*(_QWORD *)(a1 + 192) + v5);
      v10 = *(__int64 (**)())(*(_QWORD *)v9 + 144LL);
      if ( v10 == sub_2FEDB60 )
        goto LABEL_8;
      v5 += 8;
      v6 |= ((__int64 (__fastcall *)(__int64, __int64 **))v10)(v9, v51);
      if ( v7 == v5 )
      {
LABEL_12:
        v55 = v6;
        goto LABEL_13;
      }
    }
  }
  v55 = 0;
LABEL_13:
  v71[0] = 0;
  v71[1] = 0;
  v11 = (__int64)v51[7];
  v74 = 0;
  v72 = 0;
  v73 = 0;
  v75 = 0;
  v76 = 0;
  v77 = 0;
  src = 0;
  v79 = 0;
  v80 = 0;
  v81 = 0;
  v82 = 0;
  v83 = 0;
  sub_D126D0((__int64)v71, v11);
  sub_D12BD0((__int64)v71);
  v12 = 0;
  v13 = v79;
  v14 = (char *)src;
  v67[0] = v51;
  v67[1] = v71;
  v15 = a1 + 176;
  dest = 0;
  v69 = 0;
  v70 = 0;
  if ( src == v79 )
  {
    v43 = *(_DWORD *)(a1 + 200);
    if ( v43 )
      goto LABEL_76;
    goto LABEL_85;
  }
  v16 = (char *)(v79 - (_BYTE *)src);
  if ( v79 == src )
  {
LABEL_43:
    v29 = (char *)v69;
    v30 = (char *)((_BYTE *)v69 - (_BYTE *)v12);
    if ( (unsigned __int64)v16 > (_BYTE *)v69 - (_BYTE *)v12 )
    {
      v42 = &v30[(_QWORD)v14];
      if ( &v30[(_QWORD)v14] != v14 )
      {
        memmove(v12, v14, (size_t)v30);
        v29 = (char *)v69;
      }
      if ( v42 != v13 )
        v29 = (char *)memmove(v29, v42, v13 - v42);
      v69 = &v29[v13 - v42];
    }
    else
    {
      if ( v13 != v14 )
      {
        v31 = memmove(v12, v14, (size_t)v16);
        v29 = (char *)v69;
        v12 = v31;
      }
      v32 = &v16[(_QWORD)v12];
      if ( v32 != v29 )
        v69 = v32;
    }
  }
  else
  {
LABEL_15:
    if ( (unsigned __int64)v16 > 0x7FFFFFFFFFFFFFF8LL )
      sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
    v17 = (void *)sub_22077B0((unsigned __int64)v16);
    v18 = v17;
    if ( v13 != v14 )
      memcpy(v17, v14, (size_t)v16);
    if ( dest )
      j_j___libc_free_0((unsigned __int64)dest);
    dest = v18;
    v69 = &v16[(_QWORD)v18];
    v70 = &v16[(_QWORD)v18];
  }
  sub_D12BD0((__int64)v71);
  v54 = 0;
  do
  {
    v19 = *(unsigned int *)(a1 + 200);
    if ( !(_DWORD)v19 )
      break;
    v58 = 1;
    v52 = 8 * v19;
    v60 = 0;
    v56 = 0;
    v57 = 0;
    do
    {
      v20 = *(__int64 **)(*(_QWORD *)(a1 + 192) + v60);
      if ( sub_B80690() )
      {
        v64 = v66;
        v65 = 0;
        LOBYTE(v66[0]) = 0;
        sub_B817B0(v15, (__int64)v20, 0, 7, v66, 0);
        if ( v64 != v66 )
          j_j___libc_free_0((unsigned __int64)v64);
      }
      sub_B86470(v15, v20);
      sub_B89740(v15, v20);
      if ( (*(__int64 (__fastcall **)(__int64 *))(*v20 + 120))(v20) )
      {
        v61 = (char *)v69;
        if ( dest != v69 )
        {
          v21 = (char *)dest;
          v22 = 0;
          do
          {
            v24 = *(_QWORD *)(*(_QWORD *)v21 + 8LL);
            if ( v24 )
            {
              v25 = sub_BD5D20(*(_QWORD *)(*(_QWORD *)v21 + 8LL));
              sub_B817B0(v15, (__int64)v20, 0, 3, v25, v26);
              v27 = sub_BC4450(v20);
              if ( v27 )
              {
                v59 = (__int64)v27;
                sub_C9E250((__int64)v27);
                v22 |= sub_B89FF0((__int64)v20, v24);
                sub_C9E2A0(v59);
              }
              else
              {
                v22 |= sub_B89FF0((__int64)v20, v24);
              }
              v23 = sub_B2BE50(v24);
              sub_B6EAA0(v23);
            }
            v21 += 8;
          }
          while ( v61 != v21 );
          if ( (_BYTE)v22 )
          {
            v58 = 0;
LABEL_36:
            sub_B817B0(v15, (__int64)v20, 1, 7, byte_3F871B3, 0);
            sub_B865A0(v15, v20);
            nullsub_76();
            sub_B887D0(v15, v20);
            v57 = v22;
            goto LABEL_37;
          }
        }
      }
      else
      {
        v33 = *v51;
        if ( !v58 )
          v56 |= sub_30A4D70((__int64)v67, v51);
        v64 = 0;
        v65 = 0;
        v66[0] = 0x1000000000LL;
        v34 = sub_B6F970(*v33);
        v62 = (*(__int64 (__fastcall **)(__int64, const char *, __int64))(*(_QWORD *)v34 + 24LL))(v34, "size-info", 9);
        v35 = sub_BC4450(v20);
        v36 = (__int64)v35;
        if ( v35 )
          sub_C9E250((__int64)v35);
        if ( v62 )
        {
          v63 = sub_B806A0(v15, (__int64)v33, (__int64)&v64);
          v58 = (*(__int64 (__fastcall **)(__int64 *, _QWORD *))(*v20 + 152))(v20, v67);
          v41 = sub_BAA3C0((__int64)v33);
          if ( v63 != v41 )
            sub_B82CC0(v15, (__int64)v20, (__int64)v33, v41 - (unsigned __int64)v63, v63, (__int64)&v64, 0);
        }
        else
        {
          v58 = (*(__int64 (__fastcall **)(__int64 *, _QWORD *))(*v20 + 152))(v20, v67);
        }
        if ( v36 )
          sub_C9E2A0(v36);
        v37 = (unsigned __int64)v64;
        if ( HIDWORD(v65) && (_DWORD)v65 )
        {
          v38 = 8LL * (unsigned int)v65;
          v39 = 0;
          do
          {
            v40 = *(_QWORD **)(v37 + v39);
            if ( v40 != (_QWORD *)-8LL && v40 )
            {
              sub_C7D6A0((__int64)v40, *v40 + 17LL, 8);
              v37 = (unsigned __int64)v64;
            }
            v39 += 8;
          }
          while ( v38 != v39 );
        }
        _libc_free(v37);
        v57 |= v58;
        if ( v58 )
        {
          LOBYTE(v22) = v57;
          goto LABEL_36;
        }
        v58 = 1;
      }
      sub_B865A0(v15, v20);
      nullsub_76();
LABEL_37:
      sub_B87180(v15, (__int64)v20);
      sub_B81BF0(v15, (__int64)v20, byte_3F871B3, 0, 7);
      v60 += 8;
    }
    while ( v52 != v60 );
    if ( !v58 )
      v56 |= sub_30A4D70((__int64)v67, v51);
    v55 |= v57;
    v28 = unk_502E168 > v54++;
  }
  while ( (v28 & v56) != 0 );
  v13 = v79;
  v14 = (char *)src;
  if ( v79 != src )
  {
    v12 = dest;
    v16 = (char *)(v79 - (_BYTE *)src);
    if ( v79 - (_BYTE *)src > (unsigned __int64)(v70 - (_BYTE *)dest) )
      goto LABEL_15;
    goto LABEL_43;
  }
  v43 = *(_DWORD *)(a1 + 200);
  if ( v43 )
  {
LABEL_76:
    v44 = 0;
    v45 = 0;
    while ( 1 )
    {
      while ( 1 )
      {
        v46 = *(_QWORD *)(*(_QWORD *)(a1 + 192) + 8LL * v45);
        v47 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v46 + 120LL))(v46);
        if ( !v47 )
          break;
        v44 |= (*(__int64 (__fastcall **)(__int64, __int64 *))(*(_QWORD *)(v47 - 176) + 32LL))(v47 - 176, *v51);
LABEL_78:
        if ( v43 == ++v45 )
          goto LABEL_82;
      }
      v48 = *(_QWORD *)(*(_QWORD *)(a1 + 192) + 8LL * v45);
      v49 = *(__int64 (**)())(*(_QWORD *)v48 + 160LL);
      if ( v49 == sub_2FEDB70 )
        goto LABEL_78;
      ++v45;
      v44 |= ((__int64 (__fastcall *)(__int64, __int64 **))v49)(v48, v51);
      if ( v43 == v45 )
      {
LABEL_82:
        v55 |= v44;
        break;
      }
    }
  }
  if ( dest )
    j_j___libc_free_0((unsigned __int64)dest);
LABEL_85:
  if ( v81 )
    j_j___libc_free_0(v81);
  if ( src )
    j_j___libc_free_0((unsigned __int64)src);
  if ( v75 )
    j_j___libc_free_0(v75);
  sub_C7D6A0(v72, 16LL * (unsigned int)v74, 8);
  return v55;
}
