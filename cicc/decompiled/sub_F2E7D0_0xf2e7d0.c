// Function: sub_F2E7D0
// Address: 0xf2e7d0
//
__int64 __fastcall sub_F2E7D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v7; // r15
  __int64 v9; // rax
  __int64 v10; // rdi
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // r8
  int v14; // r11d
  unsigned int i; // eax
  __int64 v16; // rcx
  unsigned int v17; // eax
  __int64 v18; // rcx
  __int64 *v19; // rax
  __int64 v20; // rdx
  __int64 v21; // r8
  int v22; // r11d
  unsigned int j; // eax
  __int64 v24; // rsi
  unsigned int v25; // eax
  __int64 v26; // rax
  void *v27; // r9
  __int64 v28; // rdx
  __int64 v29; // rcx
  void **v30; // rax
  int v31; // eax
  void **v32; // rax
  unsigned int v33; // esi
  char v34; // al
  void **v35; // rax
  __int64 *v36; // rax
  __int64 v37; // rdi
  __int64 v38; // rsi
  __int64 v39; // [rsp+8h] [rbp-138h]
  __int64 v40; // [rsp+10h] [rbp-130h]
  __int64 v41; // [rsp+18h] [rbp-128h]
  __int64 v42; // [rsp+20h] [rbp-120h]
  __int64 v43; // [rsp+28h] [rbp-118h]
  __int64 v44; // [rsp+30h] [rbp-110h]
  __int64 v45; // [rsp+38h] [rbp-108h]
  __int64 v46; // [rsp+40h] [rbp-100h]
  void *v47; // [rsp+48h] [rbp-F8h]
  void *v48; // [rsp+50h] [rbp-F0h]
  _BYTE v50[16]; // [rsp+60h] [rbp-E0h] BYREF
  void (__fastcall *v51)(_BYTE *, _BYTE *, __int64); // [rsp+70h] [rbp-D0h]
  __int64 v52; // [rsp+80h] [rbp-C0h] BYREF
  __int64 v53; // [rsp+88h] [rbp-B8h]
  __int64 v54; // [rsp+90h] [rbp-B0h] BYREF
  unsigned int v55; // [rsp+98h] [rbp-A8h]
  char v56; // [rsp+9Ch] [rbp-A4h]
  _BYTE v57[16]; // [rsp+A0h] [rbp-A0h] BYREF
  __int64 v58; // [rsp+B0h] [rbp-90h] BYREF
  void **v59; // [rsp+B8h] [rbp-88h]
  __int64 v60; // [rsp+C0h] [rbp-80h]
  unsigned int v61; // [rsp+C8h] [rbp-78h]
  char v62; // [rsp+CCh] [rbp-74h]
  _BYTE v63[64]; // [rsp+D0h] [rbp-70h] BYREF
  char v64; // [rsp+110h] [rbp-30h] BYREF

  v7 = sub_BC1CD0(a4, &unk_4F8ED68, a3) + 8;
  v47 = (void *)(a1 + 32);
  v48 = (void *)(a1 + 80);
  if ( (unsigned __int8)sub_1026CA0(v7, &unk_4F8AED1, 0) )
  {
    *(_QWORD *)(a1 + 8) = a1 + 32;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 56) = a1 + 80;
LABEL_3:
    *(_BYTE *)(a1 + 76) = 1;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &unk_4F82400;
    *(_QWORD *)a1 = 1;
    return a1;
  }
  v40 = sub_BC1CD0(a4, &unk_4F86630, a3) + 8;
  v41 = sub_BC1CD0(a4, &unk_4F81450, a3) + 8;
  v42 = sub_BC1CD0(a4, &unk_4F6D3F8, a3) + 8;
  v43 = sub_BC1CD0(a4, &unk_4F8FAE8, a3) + 8;
  v44 = sub_BC1CD0(a4, &unk_4F89C30, a3) + 8;
  v45 = sub_BC1CD0(a4, &unk_4F86540, a3) + 8;
  v9 = sub_BC1CD0(a4, &unk_4F82410, a3);
  v10 = *(_QWORD *)(a3 + 40);
  v11 = *(_QWORD *)(v9 + 8);
  v12 = *(unsigned int *)(v11 + 88);
  v13 = *(_QWORD *)(v11 + 72);
  if ( !(_DWORD)v12 )
    goto LABEL_67;
  v14 = 1;
  for ( i = (v12 - 1)
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4)
              | ((unsigned __int64)(((unsigned int)&unk_4F87C68 >> 9) ^ ((unsigned int)&unk_4F87C68 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4)))); ; i = (v12 - 1) & v17 )
  {
    v16 = v13 + 24LL * i;
    if ( *(_UNKNOWN **)v16 == &unk_4F87C68 && v10 == *(_QWORD *)(v16 + 8) )
      break;
    if ( *(_QWORD *)v16 == -4096 && *(_QWORD *)(v16 + 8) == -4096 )
      goto LABEL_67;
    v17 = v14 + i;
    ++v14;
  }
  if ( v16 == v13 + 24 * v12 )
  {
LABEL_67:
    v46 = 0;
    v18 = 0;
  }
  else
  {
    v18 = *(_QWORD *)(*(_QWORD *)(v16 + 16) + 24LL);
    if ( v18 )
    {
      v53 = 1;
      v46 = v18 + 8;
      v19 = &v54;
      do
      {
        *v19 = -4096;
        v19 += 2;
      }
      while ( v19 != (__int64 *)&v64 );
      if ( (v53 & 1) == 0 )
      {
        v39 = v18;
        sub_C7D6A0(v54, 16LL * v55, 8);
        v18 = v39;
      }
      v18 = *(_QWORD *)(v18 + 16);
      if ( v18 )
        v18 = sub_BC1CD0(a4, &unk_4F8D9A8, a3) + 8;
    }
    else
    {
      v46 = 0;
    }
  }
  v20 = *(unsigned int *)(a4 + 88);
  v21 = *(_QWORD *)(a4 + 72);
  if ( !(_DWORD)v20 )
    goto LABEL_65;
  v22 = 1;
  for ( j = (v20 - 1)
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
              | ((unsigned __int64)(((unsigned int)&unk_4F8E5A8 >> 9) ^ ((unsigned int)&unk_4F8E5A8 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)))); ; j = (v20 - 1) & v25 )
  {
    v24 = v21 + 24LL * j;
    if ( *(_UNKNOWN **)v24 == &unk_4F8E5A8 && a3 == *(_QWORD *)(v24 + 8) )
      break;
    if ( *(_QWORD *)v24 == -4096 && *(_QWORD *)(v24 + 8) == -4096 )
      goto LABEL_65;
    v25 = v22 + j;
    ++v22;
  }
  if ( v24 == v21 + 24 * v20 )
  {
LABEL_65:
    v26 = 0;
  }
  else
  {
    v26 = *(_QWORD *)(*(_QWORD *)(v24 + 16) + 24LL);
    if ( v26 )
      v26 += 8;
  }
  if ( !(unsigned __int8)sub_F2DD30(a3, a2, v45, v40, v42, v44, v41, v43, v18, v26, v46, (char *)(a2 + 2272)) )
  {
    v54 = 0;
    sub_1026FC0(v7, &unk_4F8AED1, 0, &v52);
    if ( v54 )
      ((void (__fastcall *)(__int64 *, __int64 *, __int64))v54)(&v52, &v52, 3);
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 8) = v47;
    *(_QWORD *)(a1 + 56) = v48;
    goto LABEL_3;
  }
  v62 = 1;
  v52 = 0;
  v53 = (__int64)v57;
  v54 = 2;
  v55 = 0;
  v56 = 1;
  v58 = 0;
  v59 = (void **)v63;
  v60 = 2;
  v61 = 0;
  v51 = 0;
  sub_1026FC0(v7, &unk_4F8AED1, 1, v50);
  if ( v51 )
    v51(v50, v50, 3);
  if ( v62 )
  {
    v28 = (__int64)&v59[HIDWORD(v60)];
    v29 = HIDWORD(v60);
    v30 = v59;
    if ( v59 == (void **)v28 )
    {
LABEL_58:
      v31 = v61;
    }
    else
    {
      while ( *v30 != &unk_4F8ED68 )
      {
        if ( (void **)v28 == ++v30 )
          goto LABEL_58;
      }
      --HIDWORD(v60);
      v28 = (__int64)v59[HIDWORD(v60)];
      *v30 = (void *)v28;
      v29 = HIDWORD(v60);
      ++v58;
      v31 = v61;
    }
  }
  else
  {
    v36 = sub_C8CA60((__int64)&v58, (__int64)&unk_4F8ED68);
    if ( v36 )
    {
      *v36 = -2;
      ++v58;
      v29 = HIDWORD(v60);
      v31 = ++v61;
    }
    else
    {
      v29 = HIDWORD(v60);
      v31 = v61;
    }
  }
  if ( (_DWORD)v29 != v31 )
  {
LABEL_41:
    if ( !v56 )
    {
LABEL_57:
      sub_C8CC70((__int64)&v52, (__int64)&unk_4F8ED68, v28, v29, (__int64)&v52, (__int64)v27);
      goto LABEL_46;
    }
    v28 = HIDWORD(v54);
    v32 = (void **)v53;
    v29 = v53 + 8LL * HIDWORD(v54);
    v33 = HIDWORD(v54);
    if ( v29 != v53 )
      goto LABEL_45;
    goto LABEL_56;
  }
  if ( !v56 )
  {
    if ( sub_C8CA60((__int64)&v52, (__int64)&unk_4F82400) )
      goto LABEL_46;
    goto LABEL_41;
  }
  v28 = v53;
  v37 = v53 + 8LL * HIDWORD(v54);
  v33 = HIDWORD(v54);
  v32 = (void **)v53;
  v29 = v53;
  if ( v53 != v37 )
  {
    v27 = &unk_4F82400;
    while ( *(_UNKNOWN **)v29 != &unk_4F82400 )
    {
      v29 += 8;
      if ( v37 == v29 )
      {
LABEL_45:
        while ( *v32 != &unk_4F8ED68 )
        {
          if ( ++v32 == (void **)v29 )
            goto LABEL_56;
        }
        goto LABEL_46;
      }
    }
    goto LABEL_73;
  }
LABEL_56:
  if ( v33 >= (unsigned int)v54 )
    goto LABEL_57;
  HIDWORD(v54) = v33 + 1;
  *(_QWORD *)v29 = &unk_4F8ED68;
  ++v52;
LABEL_46:
  v34 = v56;
  v28 = v61;
  if ( HIDWORD(v60) == v61 )
  {
    if ( !v56 )
    {
      if ( sub_C8CA60((__int64)&v52, (__int64)&unk_4F82400) )
        goto LABEL_52;
      v34 = v56;
      goto LABEL_47;
    }
    v28 = v53;
    v33 = HIDWORD(v54);
LABEL_73:
    v35 = (void **)v28;
    v29 = v33;
    v38 = v28 + 8LL * v33;
    if ( v28 != v38 )
    {
      v27 = &unk_4F82400;
      while ( *(_UNKNOWN **)v28 != &unk_4F82400 )
      {
        v28 += 8;
        if ( v38 == v28 )
        {
LABEL_51:
          while ( *v35 != &unk_4F82408 )
          {
            if ( ++v35 == (void **)v28 )
              goto LABEL_61;
          }
          goto LABEL_52;
        }
      }
      goto LABEL_52;
    }
    goto LABEL_61;
  }
LABEL_47:
  if ( !v34 )
  {
LABEL_63:
    sub_C8CC70((__int64)&v52, (__int64)&unk_4F82408, v28, v29, (__int64)&v52, (__int64)v27);
    goto LABEL_52;
  }
  v35 = (void **)v53;
  v29 = HIDWORD(v54);
  v28 = v53 + 8LL * HIDWORD(v54);
  if ( v28 != v53 )
    goto LABEL_51;
LABEL_61:
  if ( (unsigned int)v29 >= (unsigned int)v54 )
    goto LABEL_63;
  HIDWORD(v54) = v29 + 1;
  *(_QWORD *)v28 = &unk_4F82408;
  ++v52;
LABEL_52:
  sub_C8CF70(a1, v47, 2, (__int64)v57, (__int64)&v52);
  sub_C8CF70(a1 + 48, v48, 2, (__int64)v63, (__int64)&v58);
  if ( !v62 )
    _libc_free(v59, v48);
  if ( !v56 )
    _libc_free(v53, v48);
  return a1;
}
