// Function: sub_2ECBE10
// Address: 0x2ecbe10
//
__int64 __fastcall sub_2ECBE10(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4)
{
  _QWORD *v7; // r15
  _QWORD *v8; // r14
  unsigned __int64 v9; // rsi
  _QWORD *v10; // rax
  _QWORD *v11; // rdi
  __int64 v12; // rcx
  __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  _QWORD *v17; // r9
  __int64 v18; // rax
  _QWORD *v19; // rdi
  __int64 v20; // r15
  __int64 v21; // r14
  __int64 v22; // rax
  __int64 v23; // rax
  _QWORD *v24; // rdi
  __int64 v25; // rdx
  __int64 **v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // r8
  void *v30; // r9
  void **v31; // rax
  void **v32; // rdx
  __int64 v33; // rcx
  void **v34; // rax
  int v35; // eax
  void **v36; // rax
  __int64 **v37; // rdx
  __int64 v38; // rcx
  void **v39; // rax
  int v40; // eax
  void **v41; // rax
  __int64 v42; // rcx
  __int64 v43; // r8
  __int64 v44; // r9
  __int64 *v45; // rax
  __int64 *v46; // rax
  void **v47; // rdi
  __int64 **v48; // rdi
  __int64 **v49; // rdi
  __int64 *v50; // rax
  __int64 *v51; // rax
  __int64 v52; // [rsp+0h] [rbp-A0h]
  __int64 v54; // [rsp+10h] [rbp-90h] BYREF
  void **v55; // [rsp+18h] [rbp-88h]
  __int64 v56; // [rsp+20h] [rbp-80h]
  __int64 v57; // [rsp+28h] [rbp-78h]
  __int64 v58; // [rsp+40h] [rbp-60h] BYREF
  void **v59; // [rsp+48h] [rbp-58h]
  unsigned int v60; // [rsp+54h] [rbp-4Ch]
  int v61; // [rsp+58h] [rbp-48h]
  char v62; // [rsp+5Ch] [rbp-44h]

  v7 = sub_C52410();
  v8 = v7 + 1;
  v9 = sub_C959E0();
  v10 = (_QWORD *)v7[2];
  if ( v10 )
  {
    v11 = v7 + 1;
    do
    {
      while ( 1 )
      {
        v12 = v10[2];
        v13 = v10[3];
        if ( v9 <= v10[4] )
          break;
        v10 = (_QWORD *)v10[3];
        if ( !v13 )
          goto LABEL_6;
      }
      v11 = v10;
      v10 = (_QWORD *)v10[2];
    }
    while ( v12 );
LABEL_6:
    if ( v8 != v11 && v9 >= v11[4] )
      v8 = v11;
  }
  if ( v8 != (_QWORD *)((char *)sub_C52410() + 8) )
  {
    v18 = v8[7];
    v17 = v8 + 6;
    if ( v18 )
    {
      v9 = (unsigned int)dword_5020C88;
      v19 = v8 + 6;
      do
      {
        while ( 1 )
        {
          v15 = *(_QWORD *)(v18 + 16);
          v14 = *(_QWORD *)(v18 + 24);
          if ( *(_DWORD *)(v18 + 32) >= dword_5020C88 )
            break;
          v18 = *(_QWORD *)(v18 + 24);
          if ( !v14 )
            goto LABEL_15;
        }
        v19 = (_QWORD *)v18;
        v18 = *(_QWORD *)(v18 + 16);
      }
      while ( v15 );
LABEL_15:
      if ( v17 != v19 && dword_5020C88 >= *((_DWORD *)v19 + 8) && *((_DWORD *)v19 + 9) )
      {
        if ( (_BYTE)qword_5020D08 )
          goto LABEL_19;
LABEL_24:
        *(_QWORD *)(a1 + 48) = 0;
        *(_QWORD *)(a1 + 8) = a1 + 32;
        *(_QWORD *)(a1 + 56) = a1 + 80;
        goto LABEL_21;
      }
    }
  }
  if ( !(*(unsigned __int8 (__fastcall **)(_QWORD, unsigned __int64, __int64, __int64, __int64, _QWORD *))(**(_QWORD **)(a3 + 16) + 256LL))(
          *(_QWORD *)(a3 + 16),
          v9,
          v14,
          v15,
          v16,
          v17) )
    goto LABEL_24;
LABEL_19:
  v20 = sub_2EB2140(a4, &qword_50208B0, a3);
  v21 = sub_2EB2140(a4, qword_501FE48, a3) + 8;
  v22 = sub_2EB2140(a4, &qword_50209D0, a3);
  v52 = sub_BC1CD0(*(_QWORD *)(v22 + 8), &unk_4F86540, *(_QWORD *)a3);
  v23 = sub_2EB2140(a4, (__int64 *)&unk_501EAD0, a3);
  *(_QWORD *)(*a2 + 72LL) = a4;
  v54 = v20 + 8;
  v24 = (_QWORD *)*a2;
  v55 = (void **)v21;
  v56 = v52 + 8;
  v25 = a2[1];
  v57 = v23 + 8;
  if ( !(unsigned __int8)sub_2ECB9C0(v24, a3, v25, &v54) )
  {
    *(_QWORD *)(a1 + 8) = a1 + 32;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 56) = a1 + 80;
LABEL_21:
    *(_QWORD *)(a1 + 64) = 2;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
    return a1;
  }
  sub_2EAFFB0((__int64)&v54);
  v30 = &unk_501EAD0;
  if ( v60 != v61 )
    goto LABEL_26;
  if ( BYTE4(v57) )
  {
    v31 = v55;
    v48 = (__int64 **)&v55[HIDWORD(v56)];
    v28 = HIDWORD(v56);
    v27 = (__int64 **)v55;
    if ( v55 == (void **)v48 )
      goto LABEL_71;
    while ( *v27 != &qword_4F82400 )
    {
      if ( v48 == ++v27 )
      {
LABEL_30:
        while ( *v31 != &unk_4F82408 )
        {
          if ( ++v31 == (void **)v27 )
            goto LABEL_71;
        }
        break;
      }
    }
LABEL_31:
    if ( v62 )
      goto LABEL_32;
LABEL_60:
    v45 = sub_C8CA60((__int64)&v58, (__int64)&unk_5025C20);
    v30 = &unk_501EAD0;
    if ( v45 )
    {
      *v45 = -2;
      ++v58;
      v33 = v60;
      v35 = ++v61;
    }
    else
    {
      v33 = v60;
      v35 = v61;
    }
    goto LABEL_37;
  }
  v50 = sub_C8CA60((__int64)&v54, (__int64)&qword_4F82400);
  v30 = &unk_501EAD0;
  if ( v50 )
    goto LABEL_31;
LABEL_26:
  if ( !BYTE4(v57) )
    goto LABEL_59;
  v31 = v55;
  v28 = HIDWORD(v56);
  v27 = (__int64 **)&v55[HIDWORD(v56)];
  if ( v55 != (void **)v27 )
    goto LABEL_30;
LABEL_71:
  if ( (unsigned int)v56 > (unsigned int)v28 )
  {
    HIDWORD(v56) = v28 + 1;
    *v27 = (__int64 *)&unk_4F82408;
    ++v54;
    goto LABEL_31;
  }
LABEL_59:
  sub_C8CC70((__int64)&v54, (__int64)&unk_4F82408, (__int64)v27, v28, v29, (__int64)&unk_501EAD0);
  v30 = &unk_501EAD0;
  if ( !v62 )
    goto LABEL_60;
LABEL_32:
  v32 = &v59[v60];
  v33 = v60;
  if ( v59 == v32 )
  {
LABEL_65:
    v35 = v61;
  }
  else
  {
    v34 = v59;
    while ( *v34 != &unk_5025C20 )
    {
      if ( v32 == ++v34 )
        goto LABEL_65;
    }
    v32 = (void **)v59[--v60];
    *v34 = v32;
    v33 = v60;
    ++v58;
    v35 = v61;
  }
LABEL_37:
  if ( (_DWORD)v33 != v35 )
  {
LABEL_38:
    if ( !BYTE4(v57) )
    {
LABEL_70:
      sub_C8CC70((__int64)&v54, (__int64)&unk_5025C20, (__int64)v32, v33, v29, (__int64)&unk_501EAD0);
      v30 = &unk_501EAD0;
      goto LABEL_43;
    }
    v36 = v55;
    v33 = HIDWORD(v56);
    v32 = &v55[HIDWORD(v56)];
    if ( v55 != v32 )
      goto LABEL_42;
LABEL_68:
    if ( (unsigned int)v33 < (unsigned int)v56 )
    {
      HIDWORD(v56) = v33 + 1;
      *v32 = &unk_5025C20;
      ++v54;
      goto LABEL_43;
    }
    goto LABEL_70;
  }
  if ( BYTE4(v57) )
  {
    v36 = v55;
    v47 = &v55[HIDWORD(v56)];
    v33 = HIDWORD(v56);
    v32 = v55;
    if ( v55 == v47 )
      goto LABEL_68;
    while ( *v32 != &qword_4F82400 )
    {
      if ( v47 == ++v32 )
      {
LABEL_42:
        while ( *v36 != &unk_5025C20 )
        {
          if ( ++v36 == v32 )
            goto LABEL_68;
        }
        break;
      }
    }
  }
  else
  {
    v51 = sub_C8CA60((__int64)&v54, (__int64)&qword_4F82400);
    v30 = &unk_501EAD0;
    if ( !v51 )
      goto LABEL_38;
  }
LABEL_43:
  if ( v62 )
  {
    v37 = (__int64 **)&v59[v60];
    v38 = v60;
    v39 = v59;
    if ( v59 == (void **)v37 )
      goto LABEL_64;
    while ( *v39 != &unk_501EAD0 )
    {
      if ( v37 == (__int64 **)++v39 )
        goto LABEL_64;
    }
    v37 = (__int64 **)v59[--v60];
    *v39 = v37;
    v38 = v60;
    ++v58;
    v40 = v61;
  }
  else
  {
    v46 = sub_C8CA60((__int64)&v58, (__int64)&unk_501EAD0);
    v30 = &unk_501EAD0;
    if ( !v46 )
    {
      v38 = v60;
LABEL_64:
      v40 = v61;
      goto LABEL_49;
    }
    *v46 = -2;
    ++v58;
    v38 = v60;
    v40 = ++v61;
  }
LABEL_49:
  if ( v40 == (_DWORD)v38 )
  {
    if ( BYTE4(v57) )
    {
      v41 = v55;
      v49 = (__int64 **)&v55[HIDWORD(v56)];
      v38 = HIDWORD(v56);
      v37 = (__int64 **)v55;
      if ( v55 != (void **)v49 )
      {
        while ( *v37 != &qword_4F82400 )
        {
          if ( v49 == ++v37 )
          {
LABEL_54:
            while ( *v41 != &unk_501EAD0 )
            {
              if ( ++v41 == (void **)v37 )
                goto LABEL_73;
            }
            goto LABEL_55;
          }
        }
        goto LABEL_55;
      }
      goto LABEL_73;
    }
    if ( sub_C8CA60((__int64)&v54, (__int64)&qword_4F82400) )
      goto LABEL_55;
    v30 = &unk_501EAD0;
  }
  if ( !BYTE4(v57) )
    goto LABEL_62;
  v41 = v55;
  v38 = HIDWORD(v56);
  v37 = (__int64 **)&v55[HIDWORD(v56)];
  if ( v37 != (__int64 **)v55 )
    goto LABEL_54;
LABEL_73:
  if ( (unsigned int)v56 > (unsigned int)v38 )
  {
    v38 = (unsigned int)(v38 + 1);
    HIDWORD(v56) = v38;
    *v37 = (__int64 *)&unk_501EAD0;
    ++v54;
    goto LABEL_55;
  }
LABEL_62:
  sub_C8CC70((__int64)&v54, (__int64)&unk_501EAD0, (__int64)v37, v38, v29, (__int64)&unk_501EAD0);
LABEL_55:
  sub_C8CD80(a1, a1 + 32, (__int64)&v54, v38, v29, (__int64)v30);
  sub_C8CD80(a1 + 48, a1 + 80, (__int64)&v58, v42, v43, v44);
  if ( !v62 )
    _libc_free((unsigned __int64)v59);
  if ( !BYTE4(v57) )
    _libc_free((unsigned __int64)v55);
  return a1;
}
