// Function: sub_166E6B0
// Address: 0x166e6b0
//
void __fastcall sub_166E6B0(__int64 *a1, __int64 *a2)
{
  __int64 v3; // rax
  __int64 *(__fastcall *v4)(__int64 *, __int64 *); // rdx
  _BYTE *v5; // r8
  size_t v6; // r14
  _QWORD *v7; // rax
  __int64 v8; // rax
  _BYTE *v9; // r14
  size_t v10; // r15
  _BYTE *v11; // rsi
  __int64 v12; // rdx
  __int64 v13; // rdi
  _BYTE *v14; // rbx
  unsigned __int64 v15; // r12
  __int64 v16; // rdi
  __int64 v17; // rax
  _QWORD *v18; // rdi
  _BYTE *src; // [rsp+8h] [rbp-1F8h]
  size_t v20; // [rsp+18h] [rbp-1E8h] BYREF
  _QWORD *v21; // [rsp+20h] [rbp-1E0h] BYREF
  size_t v22; // [rsp+28h] [rbp-1D8h]
  _QWORD v23[2]; // [rsp+30h] [rbp-1D0h] BYREF
  _QWORD v24[2]; // [rsp+40h] [rbp-1C0h] BYREF
  _QWORD v25[2]; // [rsp+50h] [rbp-1B0h] BYREF
  void *v26; // [rsp+60h] [rbp-1A0h] BYREF
  __int64 v27; // [rsp+68h] [rbp-198h]
  __int64 *v28; // [rsp+70h] [rbp-190h] BYREF
  __int64 v29; // [rsp+78h] [rbp-188h]
  __int64 v30; // [rsp+80h] [rbp-180h] BYREF
  _QWORD *v31; // [rsp+88h] [rbp-178h]
  __int64 v32; // [rsp+90h] [rbp-170h]
  int v33; // [rsp+98h] [rbp-168h]
  __int64 v34[2]; // [rsp+A0h] [rbp-160h] BYREF
  _QWORD v35[2]; // [rsp+B0h] [rbp-150h] BYREF
  _QWORD *v36; // [rsp+C0h] [rbp-140h]
  __int64 v37; // [rsp+C8h] [rbp-138h]
  _QWORD v38[2]; // [rsp+D0h] [rbp-130h] BYREF
  __int64 v39; // [rsp+E0h] [rbp-120h]
  __int64 v40; // [rsp+E8h] [rbp-118h]
  __int64 v41; // [rsp+F0h] [rbp-110h]
  _BYTE *v42; // [rsp+F8h] [rbp-108h]
  __int64 v43; // [rsp+100h] [rbp-100h]
  _BYTE v44[248]; // [rsp+108h] [rbp-F8h] BYREF

  v3 = *a2;
  v4 = *(__int64 *(__fastcall **)(__int64 *, __int64 *))(*a2 + 24);
  if ( v4 != sub_12BD5E0 )
  {
    ((void (__fastcall *)(_QWORD **))v4)(&v21);
    goto LABEL_11;
  }
  v24[1] = 0;
  v24[0] = v25;
  LOBYTE(v25[0]) = 0;
  LODWORD(v30) = 1;
  v29 = 0;
  v28 = 0;
  v27 = 0;
  v26 = &unk_49EFBE0;
  v31 = v24;
  (*(void (__fastcall **)(__int64 *, void **))(v3 + 16))(a2, &v26);
  if ( v29 != v27 )
    sub_16E7BA0(&v26);
  v21 = v23;
  v5 = (_BYTE *)*v31;
  v6 = v31[1];
  if ( v6 + *v31 && !v5 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v20 = v31[1];
  if ( v6 > 0xF )
  {
    src = v5;
    v17 = sub_22409D0(&v21, &v20, 0);
    v5 = src;
    v21 = (_QWORD *)v17;
    v18 = (_QWORD *)v17;
    v23[0] = v20;
  }
  else
  {
    if ( v6 == 1 )
    {
      LOBYTE(v23[0]) = *v5;
      v7 = v23;
      goto LABEL_9;
    }
    if ( !v6 )
    {
      v7 = v23;
      goto LABEL_9;
    }
    v18 = v23;
  }
  memcpy(v18, v5, v6);
  v6 = v20;
  v7 = v21;
LABEL_9:
  v22 = v6;
  *((_BYTE *)v7 + v6) = 0;
  sub_16E7BC0(&v26);
  if ( (_QWORD *)v24[0] != v25 )
    j_j___libc_free_0(v24[0], v25[0] + 1LL);
LABEL_11:
  v8 = a1[1];
  v9 = v21;
  v10 = v22;
  v11 = *(_BYTE **)(v8 + 16);
  v12 = *(_QWORD *)(v8 + 24);
  v26 = 0;
  v27 = 0;
  v28 = &v30;
  if ( v11 )
  {
    sub_166DB50((__int64 *)&v28, v11, (__int64)&v11[v12]);
  }
  else
  {
    v29 = 0;
    LOBYTE(v30) = 0;
  }
  v32 = -1;
  v33 = 0;
  v34[0] = (__int64)v35;
  if ( v9 )
  {
    sub_166DB50(v34, v9, (__int64)&v9[v10]);
  }
  else
  {
    v34[1] = 0;
    LOBYTE(v35[0]) = 0;
  }
  v13 = *a1;
  v43 = 0x400000000LL;
  v36 = v38;
  v37 = 0;
  LOBYTE(v38[0]) = 0;
  v39 = 0;
  v40 = 0;
  v41 = 0;
  v42 = v44;
  sub_166DE00(v13, (__int64)&v26);
  v14 = v42;
  v15 = (unsigned __int64)&v42[48 * (unsigned int)v43];
  if ( v42 != (_BYTE *)v15 )
  {
    do
    {
      v15 -= 48LL;
      v16 = *(_QWORD *)(v15 + 16);
      if ( v16 != v15 + 32 )
        j_j___libc_free_0(v16, *(_QWORD *)(v15 + 32) + 1LL);
    }
    while ( v14 != (_BYTE *)v15 );
    v15 = (unsigned __int64)v42;
  }
  if ( (_BYTE *)v15 != v44 )
    _libc_free(v15);
  if ( v39 )
    j_j___libc_free_0(v39, v41 - v39);
  if ( v36 != v38 )
    j_j___libc_free_0(v36, v38[0] + 1LL);
  if ( (_QWORD *)v34[0] != v35 )
    j_j___libc_free_0(v34[0], v35[0] + 1LL);
  if ( v28 != &v30 )
    j_j___libc_free_0(v28, v30 + 1);
  if ( v21 != v23 )
    j_j___libc_free_0(v21, v23[0] + 1LL);
}
