// Function: sub_2A57B70
// Address: 0x2a57b70
//
void __fastcall sub_2A57B70(void *src, __int64 a2, __int64 **a3, __int64 a4, char a5)
{
  __int64 v5; // r12
  char *v9; // rcx
  __int64 *v10; // rax
  __int64 *v11; // rsi
  __int64 *v12; // rax
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r9
  unsigned __int64 *v17; // rbx
  unsigned __int64 v18; // r8
  unsigned __int64 *v19; // r13
  unsigned __int64 v20; // rdi
  unsigned __int64 *v21; // rbx
  unsigned __int64 *v22; // r15
  unsigned __int64 *v23; // rbx
  unsigned __int64 *v24; // r14
  __int64 v25; // rsi
  char *v26; // [rsp-9C8h] [rbp-9C8h] BYREF
  char *v27; // [rsp-9C0h] [rbp-9C0h]
  char *v28; // [rsp-9B8h] [rbp-9B8h]
  __int64 **v29; // [rsp-9B0h] [rbp-9B0h]
  _QWORD v30[63]; // [rsp-9A8h] [rbp-9A8h] BYREF
  __int16 v31; // [rsp-7B0h] [rbp-7B0h]
  char v32; // [rsp-7A8h] [rbp-7A8h]
  __int64 v33; // [rsp-7A0h] [rbp-7A0h]
  __int64 v34; // [rsp-798h] [rbp-798h]
  __int64 v35; // [rsp-790h] [rbp-790h]
  unsigned int v36; // [rsp-788h] [rbp-788h]
  __int64 v37; // [rsp-780h] [rbp-780h]
  __int64 v38; // [rsp-778h] [rbp-778h]
  __int64 v39; // [rsp-770h] [rbp-770h]
  unsigned int v40; // [rsp-768h] [rbp-768h]
  __int64 v41; // [rsp-760h] [rbp-760h]
  __int64 v42; // [rsp-758h] [rbp-758h]
  __int64 v43; // [rsp-750h] [rbp-750h]
  unsigned int v44; // [rsp-748h] [rbp-748h]
  unsigned __int64 *v45; // [rsp-740h] [rbp-740h]
  __int64 v46; // [rsp-738h] [rbp-738h]
  _BYTE v47[192]; // [rsp-730h] [rbp-730h] BYREF
  unsigned __int64 *v48; // [rsp-670h] [rbp-670h]
  __int64 v49; // [rsp-668h] [rbp-668h]
  _BYTE v50[192]; // [rsp-660h] [rbp-660h] BYREF
  unsigned __int64 *v51; // [rsp-5A0h] [rbp-5A0h]
  __int64 v52; // [rsp-598h] [rbp-598h]
  _QWORD v53[129]; // [rsp-590h] [rbp-590h] BYREF
  _QWORD *v54; // [rsp-188h] [rbp-188h]
  __int64 v55; // [rsp-180h] [rbp-180h]
  int v56; // [rsp-178h] [rbp-178h]
  char v57; // [rsp-174h] [rbp-174h]
  _QWORD v58[9]; // [rsp-170h] [rbp-170h] BYREF
  __int64 *v59; // [rsp-128h] [rbp-128h]
  __int64 v60; // [rsp-120h] [rbp-120h]
  int v61; // [rsp-118h] [rbp-118h]
  char v62; // [rsp-114h] [rbp-114h]
  __int64 v63; // [rsp-110h] [rbp-110h] BYREF
  _DWORD *v64; // [rsp-D0h] [rbp-D0h]
  __int64 v65; // [rsp-C8h] [rbp-C8h]
  _DWORD v66[14]; // [rsp-C0h] [rbp-C0h] BYREF
  _BYTE *v67; // [rsp-88h] [rbp-88h]
  __int64 v68; // [rsp-80h] [rbp-80h]
  _BYTE v69[120]; // [rsp-78h] [rbp-78h] BYREF

  if ( !a2 )
    return;
  v5 = 8 * a2;
  v26 = 0;
  v27 = 0;
  v28 = 0;
  if ( (unsigned __int64)(8 * a2) > 0x7FFFFFFFFFFFFFF8LL )
    sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
  v9 = 0;
  if ( v5 )
  {
    v26 = (char *)sub_22077B0(8 * a2);
    v28 = &v26[v5];
    memcpy(v26, src, 8 * a2);
    v9 = &v26[v5];
  }
  v10 = *a3;
  v27 = v9;
  v29 = a3;
  v11 = *(__int64 **)(*(_QWORD *)(*v10 + 72) + 40LL);
  sub_AE0470((__int64)v30, v11, 0, 0);
  v12 = *a3;
  v30[54] = a4;
  v13 = sub_AA4E30(*v12);
  v30[58] = a3;
  v30[55] = v13;
  v31 = 257;
  v45 = (unsigned __int64 *)v47;
  v46 = 0x800000000LL;
  v49 = 0x800000000LL;
  v52 = 0x800000000LL;
  v30[59] = a4;
  v32 = a5;
  v54 = v58;
  v30[56] = 0;
  v30[57] = 0;
  memset(&v30[60], 0, 24);
  v33 = 0;
  v34 = 0;
  v35 = 0;
  v36 = 0;
  v37 = 0;
  v38 = 0;
  v39 = 0;
  v40 = 0;
  v41 = 0;
  v42 = 0;
  v43 = 0;
  v44 = 0;
  v48 = (unsigned __int64 *)v50;
  v51 = v53;
  v53[128] = 0;
  v55 = 8;
  v56 = 0;
  v59 = &v63;
  v65 = 0x600000000LL;
  v57 = 1;
  v58[8] = 0;
  v60 = 8;
  v61 = 0;
  v62 = 1;
  v64 = v66;
  v66[12] = 0;
  v67 = v69;
  v68 = 0xC00000000LL;
  v69[48] = 0;
  sub_2A54B40((__int64)&v26, (__int64)v11, v14, v15, (__int64)&v26, v16);
  if ( v67 != v69 )
    _libc_free((unsigned __int64)v67);
  if ( v64 != v66 )
    _libc_free((unsigned __int64)v64);
  if ( !v62 )
  {
    _libc_free((unsigned __int64)v59);
    if ( v57 )
      goto LABEL_11;
LABEL_38:
    _libc_free((unsigned __int64)v54);
    goto LABEL_11;
  }
  if ( !v57 )
    goto LABEL_38;
LABEL_11:
  v17 = v51;
  v18 = (unsigned __int64)(unsigned int)v52 << 7;
  v19 = (unsigned __int64 *)((char *)v51 + v18);
  if ( v17 != (unsigned __int64 *)((char *)v17 + v18) )
  {
    do
    {
      v19 -= 16;
      v20 = v19[8];
      if ( (unsigned __int64 *)v20 != v19 + 10 )
        _libc_free(v20);
      if ( (unsigned __int64 *)*v19 != v19 + 2 )
        _libc_free(*v19);
    }
    while ( v17 != v19 );
    v19 = v51;
  }
  if ( v19 != v53 )
    _libc_free((unsigned __int64)v19);
  v21 = v48;
  v22 = &v48[3 * (unsigned int)v49];
  if ( v48 != v22 )
  {
    do
    {
      v22 -= 3;
      if ( (unsigned __int64 *)*v22 != v22 + 2 )
        _libc_free(*v22);
    }
    while ( v21 != v22 );
    v22 = v48;
  }
  if ( v22 != (unsigned __int64 *)v50 )
    _libc_free((unsigned __int64)v22);
  v23 = v45;
  v24 = &v45[3 * (unsigned int)v46];
  if ( v45 != v24 )
  {
    do
    {
      v24 -= 3;
      if ( (unsigned __int64 *)*v24 != v24 + 2 )
        _libc_free(*v24);
    }
    while ( v23 != v24 );
    v24 = v45;
  }
  if ( v24 != (unsigned __int64 *)v47 )
    _libc_free((unsigned __int64)v24);
  sub_C7D6A0(v42, 16LL * v44, 8);
  sub_C7D6A0(v38, 16LL * v40, 8);
  v25 = 16LL * v36;
  sub_C7D6A0(v34, v25, 8);
  sub_AE9130((__int64)v30, v25);
  if ( v26 )
    j_j___libc_free_0((unsigned __int64)v26);
}
