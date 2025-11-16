// Function: sub_B7DFC0
// Address: 0xb7dfc0
//
__int64 __fastcall sub_B7DFC0(
        __int64 a1,
        __int64 *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        char a9,
        __int64 a10,
        __int64 a11)
{
  char v13; // al
  char v15; // al
  __int64 v16; // rax
  __int64 v17; // rax
  unsigned int v18; // r14d
  __int64 v19; // rax
  __int64 v20; // r15
  bool v21; // zf
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rax
  int v25; // edx
  int v26; // ecx
  int v27; // r8d
  int v28; // r9d
  void *v29; // rbx
  _QWORD *v30; // rbx
  __int64 v31; // rdi
  __int64 v32; // rbx
  __int64 *v33; // rax
  char v34; // al
  __int64 v35; // rax
  __int64 v36; // rax
  _QWORD *v37; // rdi
  void *v41; // [rsp+20h] [rbp-90h] BYREF
  unsigned __int64 v42; // [rsp+28h] [rbp-88h] BYREF
  __int64 v43; // [rsp+30h] [rbp-80h] BYREF
  char v44; // [rsp+38h] [rbp-78h]
  unsigned int v45; // [rsp+40h] [rbp-70h] BYREF
  __int64 v46; // [rsp+48h] [rbp-68h]
  __int64 v47; // [rsp+50h] [rbp-60h] BYREF
  char v48; // [rsp+58h] [rbp-58h]
  _QWORD *v49; // [rsp+60h] [rbp-50h] BYREF
  __int64 v50; // [rsp+68h] [rbp-48h]
  __int64 v51; // [rsp+70h] [rbp-40h]

  if ( a9 || !(_BYTE)a11 || a10 )
    sub_B6E8F0(a2, 1);
  sub_B6E910(a2, a10, a11);
  if ( !a4 )
  {
    v13 = *(_BYTE *)(a1 + 8);
    *(_QWORD *)a1 = 0;
    *(_BYTE *)(a1 + 8) = v13 & 0xFC | 2;
    return a1;
  }
  sub_C2E4C0(&v43, a7, a8);
  v15 = v44;
  v44 &= ~2u;
  if ( (v15 & 1) != 0 )
  {
    v16 = v43;
    v43 = 0;
    v47 = v16 | 1;
    if ( (v16 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    {
      sub_B7D3B0((__int64 *)&v49, &v47);
      v17 = (__int64)v49;
      *(_BYTE *)(a1 + 8) |= 3u;
      *(_QWORD *)a1 = v17 & 0xFFFFFFFFFFFFFFFELL;
      if ( (v47 & 1) == 0 && (v47 & 0xFFFFFFFFFFFFFFFELL) == 0 )
        goto LABEL_13;
LABEL_55:
      sub_C63C30(&v47);
    }
    v45 = 0;
    v18 = 0;
    v46 = sub_2241E40();
  }
  else
  {
    v45 = 0;
    v46 = sub_2241E40();
    v18 = 3 * ((_DWORD)v43 == 1);
  }
  v19 = sub_22077B0(152);
  v20 = v19;
  if ( v19 )
    sub_CA0DC0(v19, a3, a4, &v45, v18);
  if ( !v45 )
  {
    if ( (v44 & 2) != 0 )
LABEL_56:
      sub_B7CE90(&v43);
    sub_C2E790(&v47, (unsigned int)v43, 0, *(_QWORD *)(v20 + 144));
    v21 = (v48 & 1) == 0;
    v22 = v47;
    v48 &= ~2u;
    if ( !v21 )
    {
      v47 = 0;
      v42 = v22 | 1;
      if ( (v22 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      {
        sub_B7D3B0((__int64 *)&v49, (__int64 *)&v42);
        goto LABEL_25;
      }
      v22 = 0;
    }
    v42 = v22;
    v50 = a4;
    v47 = 0;
    v49 = (_QWORD *)a3;
    LOBYTE(v51) = 1;
    v24 = sub_22077B0(72);
    v29 = (void *)v24;
    if ( v24 )
      sub_C2ECE0(v24, (unsigned int)&v42, v25, v26, v27, v28, (__int64)v49, v50, v51);
    v41 = v29;
    if ( v42 )
      (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v42 + 8LL))(v42);
    sub_B6E9A0(a2, (__int64 *)&v41);
    v30 = v41;
    if ( v41 )
    {
      if ( *((_BYTE *)v41 + 64) )
      {
        v37 = (_QWORD *)*((_QWORD *)v41 + 4);
        *((_BYTE *)v41 + 64) = 0;
        if ( v37 != v30 + 6 )
          j_j___libc_free_0(v37, v30[6] + 1LL);
      }
      v31 = v30[3];
      if ( v31 )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v31 + 8LL))(v31);
      if ( *((_BYTE *)v30 + 16) )
      {
        *((_BYTE *)v30 + 16) = 0;
        sub_C88FF0(v30);
      }
      j_j___libc_free_0(v30, 72);
    }
    v32 = sub_B6E990((__int64)a2);
    v33 = (__int64 *)sub_22077B0(8);
    if ( v33 )
      *v33 = v32;
    v49 = v33;
    sub_B6EA60(a2, (__int64 *)&v49);
    if ( v49 )
      j_j___libc_free_0(v49, 8);
    if ( !a6 || (v36 = sub_B6E990((__int64)a2), sub_C2ED90(&v42, v36, a5, a6), (v42 & 0xFFFFFFFFFFFFFFFELL) == 0) )
    {
      v34 = *(_BYTE *)(a1 + 8);
      *(_QWORD *)a1 = v20;
      v20 = 0;
      *(_BYTE *)(a1 + 8) = v34 & 0xFC | 2;
LABEL_27:
      if ( (v48 & 2) != 0 )
        sub_B7CF00(&v47);
      if ( v47 )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v47 + 8LL))(v47);
      goto LABEL_30;
    }
    v42 = v42 & 0xFFFFFFFFFFFFFFFELL | 1;
    sub_B7DE90((__int64 *)&v49, (__int64 *)&v42);
LABEL_25:
    v23 = (__int64)v49;
    *(_BYTE *)(a1 + 8) |= 3u;
    *(_QWORD *)a1 = v23 & 0xFFFFFFFFFFFFFFFELL;
    if ( (v42 & 1) != 0 || (v42 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      sub_C63C30(&v42);
    goto LABEL_27;
  }
  sub_C63CA0(&v47, v45, v46);
  sub_B7D920((__int64 *)&v49, &v47);
  v35 = (__int64)v49;
  *(_BYTE *)(a1 + 8) |= 3u;
  *(_QWORD *)a1 = v35 & 0xFFFFFFFFFFFFFFFELL;
  if ( (v47 & 1) != 0 || (v47 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    goto LABEL_55;
LABEL_30:
  if ( v20 )
  {
    if ( *(_BYTE *)(v20 + 136) )
    {
      *(_BYTE *)(v20 + 136) = 0;
      sub_CB5B00((void *)(v20 + 40));
    }
    sub_CA0D30(v20);
    j_j___libc_free_0(v20, 152);
  }
LABEL_13:
  if ( (v44 & 2) != 0 )
    goto LABEL_56;
  if ( (v44 & 1) != 0 && v43 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v43 + 8LL))(v43);
  return a1;
}
