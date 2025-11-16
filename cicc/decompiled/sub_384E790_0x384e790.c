// Function: sub_384E790
// Address: 0x384e790
//
__int64 __fastcall sub_384E790(__int64 a1)
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
  char *v12; // r13
  char *v13; // r12
  void *v14; // rdi
  char *v15; // rbx
  void *v16; // rax
  void *v17; // r14
  __int64 v18; // r15
  __int64 v19; // rax
  __int64 *v20; // r12
  __int64 **v21; // rbx
  char v22; // al
  unsigned __int64 v23; // rdi
  __int64 v24; // rax
  unsigned __int64 v25; // r13
  const char *v26; // rax
  size_t v27; // rdx
  __m128i *v28; // rax
  __int64 v29; // r14
  char v30; // al
  bool v31; // al
  char *v32; // rcx
  char *v33; // rdx
  void *v34; // rax
  char *v35; // rbx
  __int64 *v36; // r14
  __int64 v37; // rax
  char v38; // bl
  __m128i *v39; // rax
  __int64 v40; // r13
  char v41; // bl
  unsigned __int64 v42; // rdi
  char *v43; // r14
  int v44; // r12d
  int v45; // ebx
  unsigned int v46; // r13d
  __int64 v47; // rdi
  __int64 v48; // rax
  __int64 v49; // rdi
  __int64 (*v50)(); // rax
  unsigned int v52; // [rsp+14h] [rbp-13Ch]
  __int64 **v53; // [rsp+18h] [rbp-138h]
  __int64 v54; // [rsp+30h] [rbp-120h]
  __int64 v55; // [rsp+40h] [rbp-110h]
  __int64 **v57; // [rsp+50h] [rbp-100h]
  unsigned __int8 v58; // [rsp+58h] [rbp-F8h]
  unsigned __int8 v59; // [rsp+59h] [rbp-F7h]
  char v60; // [rsp+5Ah] [rbp-F6h]
  char v61; // [rsp+5Bh] [rbp-F5h]
  char v62; // [rsp+5Ch] [rbp-F4h]
  unsigned int v63; // [rsp+5Ch] [rbp-F4h]
  unsigned __int64 v64[2]; // [rsp+60h] [rbp-F0h] BYREF
  char v65[16]; // [rsp+70h] [rbp-E0h] BYREF
  _QWORD v66[2]; // [rsp+80h] [rbp-D0h] BYREF
  void *dest; // [rsp+90h] [rbp-C0h]
  void *v68; // [rsp+98h] [rbp-B8h]
  char *v69; // [rsp+A0h] [rbp-B0h]
  _QWORD v70[2]; // [rsp+B0h] [rbp-A0h] BYREF
  unsigned __int64 v71; // [rsp+C0h] [rbp-90h]
  __int64 v72; // [rsp+C8h] [rbp-88h]
  __int64 v73; // [rsp+D0h] [rbp-80h]
  unsigned __int64 v74; // [rsp+D8h] [rbp-78h]
  __int64 v75; // [rsp+E0h] [rbp-70h]
  __int64 v76; // [rsp+E8h] [rbp-68h]
  void *src; // [rsp+F0h] [rbp-60h]
  char *v78; // [rsp+F8h] [rbp-58h]
  __int64 v79; // [rsp+100h] [rbp-50h]
  unsigned __int64 v80; // [rsp+108h] [rbp-48h]
  __int64 v81; // [rsp+110h] [rbp-40h]
  __int64 v82; // [rsp+118h] [rbp-38h]

  v1 = *(__int64 **)(a1 + 8);
  v2 = *v1;
  v3 = v1[1];
  if ( v2 == v3 )
LABEL_97:
    BUG();
  while ( *(_UNKNOWN **)v2 != &unk_4F98A8D )
  {
    v2 += 16;
    if ( v3 == v2 )
      goto LABEL_97;
  }
  v53 = *(__int64 ***)((*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v2 + 8) + 104LL))(
                         *(_QWORD *)(v2 + 8),
                         &unk_4F98A8D)
                     + 160);
  v4 = *(unsigned int *)(a1 + 192);
  if ( (_DWORD)v4 )
  {
    v5 = 0;
    v6 = 0;
    v7 = 8 * v4;
    while ( 1 )
    {
      while ( 1 )
      {
        v8 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(*(_QWORD *)(a1 + 184) + v5) + 120LL))(*(_QWORD *)(*(_QWORD *)(a1 + 184) + v5));
        if ( !v8 )
          break;
        v6 |= (*(__int64 (__fastcall **)(__int64, __int64 *))(*(_QWORD *)(v8 - 160) + 24LL))(v8 - 160, *v53);
LABEL_8:
        v5 += 8;
        if ( v7 == v5 )
          goto LABEL_12;
      }
      v9 = *(_QWORD *)(*(_QWORD *)(a1 + 184) + v5);
      v10 = *(__int64 (**)())(*(_QWORD *)v9 + 144LL);
      if ( v10 == sub_18480F0 )
        goto LABEL_8;
      v5 += 8;
      v6 |= ((__int64 (__fastcall *)(__int64, __int64 **))v10)(v9, v53);
      if ( v7 == v5 )
      {
LABEL_12:
        v58 = v6;
        goto LABEL_13;
      }
    }
  }
  v58 = 0;
LABEL_13:
  v70[0] = 0;
  v70[1] = 0;
  v11 = (__int64)v53[7];
  v73 = 0;
  v71 = 0;
  v72 = 0;
  v74 = 0;
  v75 = 0;
  v76 = 0;
  src = 0;
  v78 = 0;
  v79 = 0;
  v80 = 0;
  v81 = 0;
  v82 = 0;
  sub_13C69A0((__int64)v70, v11);
  sub_13C6E30((__int64)v70);
  v12 = v78;
  v13 = (char *)src;
  v14 = 0;
  v66[0] = v53;
  v66[1] = v70;
  dest = 0;
  v68 = 0;
  v69 = 0;
  if ( src == v78 )
  {
    v44 = *(_DWORD *)(a1 + 192);
    if ( v44 )
      goto LABEL_77;
    goto LABEL_86;
  }
  v15 = (char *)(v78 - (_BYTE *)src);
  if ( v78 == src )
  {
LABEL_48:
    v32 = (char *)v68;
    v33 = (char *)((_BYTE *)v68 - (_BYTE *)v14);
    if ( (unsigned __int64)v15 > (_BYTE *)v68 - (_BYTE *)v14 )
    {
      v43 = &v33[(_QWORD)v13];
      if ( v13 != &v33[(_QWORD)v13] )
      {
        memmove(v14, v13, (size_t)v33);
        v32 = (char *)v68;
      }
      if ( v12 != v43 )
        v32 = (char *)memmove(v32, v43, v12 - v43);
      v68 = &v32[v12 - v43];
    }
    else
    {
      if ( v13 != v12 )
      {
        v34 = memmove(v14, v13, (size_t)v15);
        v32 = (char *)v68;
        v14 = v34;
      }
      v35 = &v15[(_QWORD)v14];
      if ( v35 != v32 )
        v68 = v35;
    }
  }
  else
  {
LABEL_15:
    if ( (unsigned __int64)v15 > 0x7FFFFFFFFFFFFFF8LL )
      sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
    v16 = (void *)sub_22077B0((unsigned __int64)v15);
    v17 = v16;
    if ( v13 != v12 )
      memcpy(v16, v13, (size_t)v15);
    if ( dest )
      j_j___libc_free_0((unsigned __int64)dest);
    dest = v17;
    v68 = &v15[(_QWORD)v17];
    v69 = &v15[(_QWORD)v17];
  }
  sub_13C6E30((__int64)v70);
  v52 = 0;
  v18 = a1 + 160;
  do
  {
    v19 = *(unsigned int *)(a1 + 192);
    if ( !(_DWORD)v19 )
      break;
    v60 = 1;
    v54 = 8 * v19;
    v55 = 0;
    v59 = 0;
    v61 = 0;
    do
    {
      v20 = *(__int64 **)(*(_QWORD *)(a1 + 184) + v55);
      if ( sub_160E750() )
      {
        v64[1] = 0;
        v64[0] = (unsigned __int64)v65;
        v65[0] = 0;
        sub_160F160(v18, (__int64)v20, 0, 8, v65, 0);
        if ( (char *)v64[0] != v65 )
          j_j___libc_free_0(v64[0]);
      }
      sub_1615D60(v18, v20);
      sub_1614C80(v18, (__int64)v20);
      if ( !(*(__int64 (__fastcall **)(__int64 *))(*v20 + 120))(v20) )
      {
        v36 = *v53;
        if ( !v60 )
          v59 |= sub_384D760((__int64)v66, v53);
        v37 = sub_16033E0(*v36);
        v38 = (*(__int64 (__fastcall **)(__int64, const char *, __int64))(*(_QWORD *)v37 + 24LL))(v37, "size-info", 9);
        v39 = (__m128i *)sub_1612E30(v20);
        v40 = (__int64)v39;
        if ( v39 )
          sub_16D7910(v39);
        sub_1403F30(v64, v20, *(_QWORD *)(a1 + 168));
        if ( v38 )
        {
          v63 = sub_160E760(v18, (__int64)v36);
          v41 = (*(__int64 (__fastcall **)(__int64 *, _QWORD *))(*v20 + 152))(v20, v66);
          sub_160FF80(v18, (__int64)v20, (__int64)v36, v63);
        }
        else
        {
          v41 = (*(__int64 (__fastcall **)(__int64 *, _QWORD *))(*v20 + 152))(v20, v66);
        }
        v42 = v64[0];
        if ( v64[0] )
        {
          if ( v41 )
          {
            (*(void (__fastcall **)(unsigned __int64, __int64))(*(_QWORD *)v64[0] + 56LL))(v64[0], 2);
            v42 = v64[0];
          }
          (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v42 + 48LL))(v42);
        }
        if ( v40 )
          sub_16D7950(v40);
        v61 |= v41;
        v60 = 1;
        goto LABEL_66;
      }
      v21 = (__int64 **)dest;
      v57 = (__int64 **)v68;
      if ( dest == v68 )
        goto LABEL_66;
      v62 = 0;
      do
      {
        v25 = **v21;
        if ( v25 )
        {
          v26 = sub_1649960(**v21);
          sub_160F160(v18, (__int64)v20, 0, 4, v26, v27);
          v28 = (__m128i *)sub_1612E30(v20);
          v29 = (__int64)v28;
          if ( v28 )
          {
            sub_16D7910(v28);
            sub_1403F30(v64, v20, *(_QWORD *)(a1 + 168));
            v22 = sub_1619FD0((__int64)v20, v25);
            v23 = v64[0];
            v62 |= v22;
            if ( !v64[0] )
            {
LABEL_32:
              sub_16D7950(v29);
LABEL_33:
              v24 = sub_15E0530(v25);
              sub_16027A0(v24);
              goto LABEL_34;
            }
          }
          else
          {
            sub_1403F30(v64, v20, *(_QWORD *)(a1 + 168));
            v30 = sub_1619FD0((__int64)v20, v25);
            v23 = v64[0];
            v62 |= v30;
            if ( !v64[0] )
              goto LABEL_33;
          }
          if ( v62 )
          {
            (*(void (__fastcall **)(unsigned __int64, __int64))(*(_QWORD *)v23 + 56LL))(v23, 2);
            v23 = v64[0];
          }
          (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v23 + 48LL))(v23);
          if ( !v29 )
            goto LABEL_33;
          goto LABEL_32;
        }
LABEL_34:
        ++v21;
      }
      while ( v57 != v21 );
      if ( v62 )
      {
        v60 = 0;
        goto LABEL_41;
      }
LABEL_66:
      if ( v61 )
      {
LABEL_41:
        sub_160F160(v18, (__int64)v20, 1, 8, byte_3F871B3, 0);
        v61 = 1;
      }
      sub_1615E90(v18, v20);
      nullsub_568();
      sub_16145F0(v18, (__int64)v20);
      sub_16176C0(v18, (__int64)v20);
      sub_1615450(v18, (__int64)v20, byte_3F871B3, 0, 8);
      v55 += 8;
    }
    while ( v54 != v55 );
    if ( !v60 )
      v59 |= sub_384D760((__int64)v66, v53);
    v58 |= v61;
    v31 = dword_5051780 > v52++;
  }
  while ( (v31 & v59) != 0 );
  v12 = v78;
  v13 = (char *)src;
  if ( v78 != src )
  {
    v14 = dest;
    v15 = (char *)(v78 - (_BYTE *)src);
    if ( v78 - (_BYTE *)src > (unsigned __int64)(v69 - (_BYTE *)dest) )
      goto LABEL_15;
    goto LABEL_48;
  }
  v44 = *(_DWORD *)(a1 + 192);
  if ( v44 )
  {
LABEL_77:
    v45 = 0;
    v46 = 0;
    while ( 1 )
    {
      while ( 1 )
      {
        v47 = *(_QWORD *)(*(_QWORD *)(a1 + 184) + 8LL * v46);
        v48 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v47 + 120LL))(v47);
        if ( !v48 )
          break;
        v45 |= (*(__int64 (__fastcall **)(__int64, __int64 *))(*(_QWORD *)(v48 - 160) + 32LL))(v48 - 160, *v53);
LABEL_79:
        if ( v44 == ++v46 )
          goto LABEL_83;
      }
      v49 = *(_QWORD *)(*(_QWORD *)(a1 + 184) + 8LL * v46);
      v50 = *(__int64 (**)())(*(_QWORD *)v49 + 160LL);
      if ( v50 == sub_18322F0 )
        goto LABEL_79;
      ++v46;
      v45 |= ((__int64 (__fastcall *)(__int64, __int64 **))v50)(v49, v53);
      if ( v44 == v46 )
      {
LABEL_83:
        v58 |= v45;
        break;
      }
    }
  }
  if ( dest )
    j_j___libc_free_0((unsigned __int64)dest);
LABEL_86:
  if ( v80 )
    j_j___libc_free_0(v80);
  if ( src )
    j_j___libc_free_0((unsigned __int64)src);
  if ( v74 )
    j_j___libc_free_0(v74);
  j___libc_free_0(v71);
  return v58;
}
