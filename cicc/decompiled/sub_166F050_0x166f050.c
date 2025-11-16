// Function: sub_166F050
// Address: 0x166f050
//
_QWORD *__fastcall sub_166F050(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        unsigned __int8 a4,
        __int8 *a5,
        size_t a6,
        __m128i a7,
        __m128i *a8,
        unsigned __int64 a9,
        __int64 a10,
        __int64 a11)
{
  __int64 v15; // rbx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 v19; // rdx
  __int64 v20; // rdx
  char v22; // al
  unsigned __int64 v23; // rax
  __int64 v24; // rdx
  unsigned __int64 v25; // rax
  char v26; // al
  __int64 *v27; // r8
  __int64 **v28; // rax
  __int64 **v29; // rcx
  __int64 v30; // rax
  __int64 **v31; // rcx
  __int64 v32; // rax
  _QWORD **v33; // rdi
  __int64 v34; // rax
  _QWORD **v35; // r13
  __int64 **v36; // [rsp+8h] [rbp-C8h]
  __int64 *v37; // [rsp+20h] [rbp-B0h]
  __int64 *v39; // [rsp+28h] [rbp-A8h]
  __int64 **v40; // [rsp+28h] [rbp-A8h]
  int v41[2]; // [rsp+38h] [rbp-98h] BYREF
  __int64 v42; // [rsp+40h] [rbp-90h] BYREF
  __int64 v43; // [rsp+48h] [rbp-88h] BYREF
  __int64 v44; // [rsp+50h] [rbp-80h] BYREF
  __int64 v45; // [rsp+58h] [rbp-78h] BYREF
  __int64 v46; // [rsp+60h] [rbp-70h] BYREF
  __int64 *v47; // [rsp+68h] [rbp-68h] BYREF
  unsigned __int64 v48; // [rsp+70h] [rbp-60h] BYREF
  unsigned __int64 v49; // [rsp+78h] [rbp-58h] BYREF
  _QWORD **v50; // [rsp+80h] [rbp-50h] BYREF
  char v51; // [rsp+88h] [rbp-48h]
  __int64 v52[8]; // [rsp+90h] [rbp-40h] BYREF

  v15 = a2;
  sub_16D8B50((int)v41, (int)"parse", 5, (int)"Parse IR", 8, unk_4F9E388, "irparse", 7u, (__int64)"LLVM IR Parsing", 15);
  if ( !a9 )
    goto LABEL_5;
  v19 = a8->m128i_u8[0];
  if ( (_BYTE)v19 != 0xDE )
  {
    if ( (_BYTE)v19 == 66 && a8->m128i_i8[1] == 67 && a8->m128i_i8[2] == -64 && a8->m128i_i8[3] == -34 )
      goto LABEL_15;
LABEL_5:
    sub_38809A0((_DWORD)a1, a2, a3, 0, a4, v18, (__int64)a8, a9, a10, a11, (__int64)a5, a6);
    goto LABEL_6;
  }
  if ( a8->m128i_i8[1] != -64 || a8->m128i_i8[2] != 23 || a8->m128i_i8[3] != 11 )
    goto LABEL_5;
LABEL_15:
  a2 = a3;
  sub_1509BC0((__int64)&v50, a3, v19, v16, v17, v18, a7, a8, a9);
  v22 = v51;
  v51 &= ~2u;
  if ( (v22 & 1) == 0 )
  {
    v33 = v50;
    goto LABEL_33;
  }
  v23 = (unsigned __int64)v50;
  v50 = 0;
  v24 = v23 | 1;
  v25 = v23 & 0xFFFFFFFFFFFFFFFELL;
  v42 = v24;
  if ( !v25 )
  {
    v33 = 0;
LABEL_33:
    v20 = a6;
    if ( a6 )
    {
      a2 = (__int64)a5;
      sub_1632B30((__int64)v33, a5, a6);
      v33 = v50;
      if ( (v51 & 2) != 0 )
        goto LABEL_44;
    }
    *a1 = v33;
    goto LABEL_6;
  }
  v52[0] = v15;
  a2 = (__int64)&unk_4FA032A;
  v52[1] = (__int64)&a8;
  v42 = 0;
  v43 = 0;
  v44 = 0;
  v39 = (__int64 *)v25;
  v26 = (*(__int64 (__fastcall **)(unsigned __int64, void *))(*(_QWORD *)v25 + 48LL))(v25, &unk_4FA032A);
  v27 = v39;
  if ( v26 )
  {
    v28 = (__int64 **)v39[2];
    v29 = (__int64 **)v39[1];
    v45 = 1;
    v36 = v28;
    if ( v29 == v28 )
    {
      v32 = 1;
    }
    else
    {
      do
      {
        v37 = v27;
        v47 = *v29;
        *v29 = 0;
        v40 = v29;
        sub_166EAE0((__int64 *)&v48, &v47, v52);
        v30 = v45;
        a2 = (__int64)&v46;
        v45 = 0;
        v46 = v30 | 1;
        sub_12BEC00(&v49, (unsigned __int64 *)&v46, &v48);
        if ( (v45 & 1) != 0 || (v31 = v40, v27 = v37, (v45 & 0xFFFFFFFFFFFFFFFELL) != 0) )
          sub_16BCAE0(&v45);
        v45 |= v49 | 1;
        if ( (v46 & 1) != 0 || (v46 & 0xFFFFFFFFFFFFFFFELL) != 0 )
          sub_16BCAE0(&v46);
        if ( (v48 & 1) != 0 || (v48 & 0xFFFFFFFFFFFFFFFELL) != 0 )
          sub_16BCAE0(&v48);
        if ( v47 )
        {
          (*(void (__fastcall **)(__int64 *))(*v47 + 8))(v47);
          v27 = v37;
          v31 = v40;
        }
        v29 = v31 + 1;
      }
      while ( v36 != v29 );
      v32 = v45 | 1;
    }
    v48 = v32;
    (*(void (__fastcall **)(__int64 *))(*v27 + 8))(v27);
  }
  else
  {
    v49 = (unsigned __int64)v39;
    a2 = (__int64)&v49;
    sub_166EAE0((__int64 *)&v48, (__int64 **)&v49, v52);
    if ( v49 )
      (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v49 + 8LL))(v49);
  }
  if ( (v48 & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    v48 = v48 & 0xFFFFFFFFFFFFFFFELL | 1;
    sub_16BCAE0(&v48);
  }
  if ( (v44 & 1) != 0 || (v44 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    sub_16BCAE0(&v44);
  if ( (v43 & 1) != 0 || (v43 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    sub_16BCAE0(&v43);
  v34 = v42;
  *a1 = 0;
  if ( (v34 & 1) != 0 || (v34 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    sub_16BCAE0(&v42);
  if ( (v51 & 2) != 0 )
LABEL_44:
    sub_1264230(&v50, a2, v20);
  v35 = v50;
  if ( (v51 & 1) != 0 )
  {
    if ( v50 )
      ((void (__fastcall *)(_QWORD **))(*v50)[1])(v50);
  }
  else if ( v50 )
  {
    sub_1633490(v50);
    a2 = 736;
    j_j___libc_free_0(v35, 736);
  }
LABEL_6:
  if ( *(_QWORD *)v41 )
    sub_16D7950(*(_QWORD *)v41, a2, v20);
  return a1;
}
