// Function: sub_166EB60
// Address: 0x166eb60
//
void __fastcall sub_166EB60(__int64 a1, __int64 *a2)
{
  __int64 v3; // rax
  __int64 *(__fastcall *v4)(__int64 *, __int64 *); // rdx
  _BYTE *v5; // r8
  size_t v6; // r14
  _QWORD *v7; // rax
  _BYTE *v8; // r15
  size_t v9; // r14
  char *(*v10)(); // rax
  char *v11; // rsi
  __int64 v12; // rdx
  __int64 v13; // rdi
  _BYTE *v14; // rbx
  unsigned __int64 v15; // r12
  __int64 v16; // rdi
  __int64 v17; // rax
  __int64 v18; // rax
  _QWORD *v19; // rdi
  _BYTE *src; // [rsp+8h] [rbp-1F8h]
  size_t v21; // [rsp+18h] [rbp-1E8h] BYREF
  _QWORD *v22; // [rsp+20h] [rbp-1E0h] BYREF
  size_t v23; // [rsp+28h] [rbp-1D8h]
  _QWORD v24[2]; // [rsp+30h] [rbp-1D0h] BYREF
  _QWORD v25[2]; // [rsp+40h] [rbp-1C0h] BYREF
  _QWORD v26[2]; // [rsp+50h] [rbp-1B0h] BYREF
  void *v27; // [rsp+60h] [rbp-1A0h] BYREF
  __int64 v28; // [rsp+68h] [rbp-198h]
  __int64 *v29; // [rsp+70h] [rbp-190h] BYREF
  __int64 v30; // [rsp+78h] [rbp-188h]
  __int64 v31; // [rsp+80h] [rbp-180h] BYREF
  _QWORD *v32; // [rsp+88h] [rbp-178h]
  __int64 v33; // [rsp+90h] [rbp-170h]
  int v34; // [rsp+98h] [rbp-168h]
  __int64 v35[2]; // [rsp+A0h] [rbp-160h] BYREF
  _QWORD v36[2]; // [rsp+B0h] [rbp-150h] BYREF
  _QWORD *v37; // [rsp+C0h] [rbp-140h]
  __int64 v38; // [rsp+C8h] [rbp-138h]
  _QWORD v39[2]; // [rsp+D0h] [rbp-130h] BYREF
  __int64 v40; // [rsp+E0h] [rbp-120h]
  __int64 v41; // [rsp+E8h] [rbp-118h]
  __int64 v42; // [rsp+F0h] [rbp-110h]
  _BYTE *v43; // [rsp+F8h] [rbp-108h]
  __int64 v44; // [rsp+100h] [rbp-100h]
  _BYTE v45[248]; // [rsp+108h] [rbp-F8h] BYREF

  v3 = *a2;
  v4 = *(__int64 *(__fastcall **)(__int64 *, __int64 *))(*a2 + 24);
  if ( v4 != sub_12BD5E0 )
  {
    ((void (__fastcall *)(_QWORD **))v4)(&v22);
    goto LABEL_11;
  }
  v25[1] = 0;
  v25[0] = v26;
  LOBYTE(v26[0]) = 0;
  LODWORD(v31) = 1;
  v30 = 0;
  v29 = 0;
  v28 = 0;
  v27 = &unk_49EFBE0;
  v32 = v25;
  (*(void (__fastcall **)(__int64 *, void **))(v3 + 16))(a2, &v27);
  if ( v30 != v28 )
    sub_16E7BA0(&v27);
  v22 = v24;
  v5 = (_BYTE *)*v32;
  v6 = v32[1];
  if ( v6 + *v32 && !v5 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v21 = v32[1];
  if ( v6 > 0xF )
  {
    src = v5;
    v18 = sub_22409D0(&v22, &v21, 0);
    v5 = src;
    v22 = (_QWORD *)v18;
    v19 = (_QWORD *)v18;
    v24[0] = v21;
  }
  else
  {
    if ( v6 == 1 )
    {
      LOBYTE(v24[0]) = *v5;
      v7 = v24;
      goto LABEL_9;
    }
    if ( !v6 )
    {
      v7 = v24;
      goto LABEL_9;
    }
    v19 = v24;
  }
  memcpy(v19, v5, v6);
  v6 = v21;
  v7 = v22;
LABEL_9:
  v23 = v6;
  *((_BYTE *)v7 + v6) = 0;
  sub_16E7BC0(&v27);
  if ( (_QWORD *)v25[0] != v26 )
    j_j___libc_free_0(v25[0], v26[0] + 1LL);
LABEL_11:
  v8 = v22;
  v9 = v23;
  v10 = *(char *(**)())(***(_QWORD ***)(a1 + 8) + 16LL);
  if ( v10 == sub_12BCB10 )
  {
    v27 = 0;
    v11 = "Unknown buffer";
    v12 = 14;
    v28 = 0;
LABEL_13:
    v29 = &v31;
    sub_166DB50((__int64 *)&v29, v11, (__int64)&v11[v12]);
    goto LABEL_14;
  }
  v17 = (__int64)v10();
  v27 = 0;
  v28 = 0;
  v11 = (char *)v17;
  if ( v17 )
    goto LABEL_13;
  LOBYTE(v31) = 0;
  v29 = &v31;
  v30 = 0;
LABEL_14:
  v33 = -1;
  v34 = 0;
  v35[0] = (__int64)v36;
  if ( v8 )
  {
    sub_166DB50(v35, v8, (__int64)&v8[v9]);
  }
  else
  {
    v35[1] = 0;
    LOBYTE(v36[0]) = 0;
  }
  v13 = *(_QWORD *)a1;
  v44 = 0x400000000LL;
  v37 = v39;
  v38 = 0;
  LOBYTE(v39[0]) = 0;
  v40 = 0;
  v41 = 0;
  v42 = 0;
  v43 = v45;
  sub_166DE00(v13, (__int64)&v27);
  v14 = v43;
  v15 = (unsigned __int64)&v43[48 * (unsigned int)v44];
  if ( v43 != (_BYTE *)v15 )
  {
    do
    {
      v15 -= 48LL;
      v16 = *(_QWORD *)(v15 + 16);
      if ( v16 != v15 + 32 )
        j_j___libc_free_0(v16, *(_QWORD *)(v15 + 32) + 1LL);
    }
    while ( v14 != (_BYTE *)v15 );
    v15 = (unsigned __int64)v43;
  }
  if ( (_BYTE *)v15 != v45 )
    _libc_free(v15);
  if ( v40 )
    j_j___libc_free_0(v40, v42 - v40);
  if ( v37 != v39 )
    j_j___libc_free_0(v37, v39[0] + 1LL);
  if ( (_QWORD *)v35[0] != v36 )
    j_j___libc_free_0(v35[0], v36[0] + 1LL);
  if ( v29 != &v31 )
    j_j___libc_free_0(v29, v31 + 1);
  if ( v22 != v24 )
    j_j___libc_free_0(v22, v24[0] + 1LL);
}
