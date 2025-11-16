// Function: sub_1AA3870
// Address: 0x1aa3870
//
_QWORD *__fastcall sub_1AA3870(_QWORD *a1, __int64 *a2)
{
  int v2; // r8d
  int v3; // r9d
  __int64 v4; // rbx
  char *v5; // rdx
  _BYTE *v6; // rax
  _QWORD *v7; // rdi
  __int64 v8; // rdi
  _BYTE *v9; // rax
  __int64 v10; // rdi
  _BYTE *v11; // rax
  __int64 v12; // rdi
  _BYTE *v13; // rax
  const char *v14; // r13
  void *v15; // rax
  size_t v16; // r15
  __int64 v17; // rsi
  _BYTE *v18; // r10
  size_t v19; // r8
  const char *v20; // rax
  char *v21; // rdi
  unsigned __int64 v22; // r12
  const void *v23; // r13
  void *v24; // rdi
  int v25; // eax
  const char *v27; // rax
  char *v28; // rdi
  size_t n; // [rsp+8h] [rbp-948h]
  _BYTE *src; // [rsp+10h] [rbp-940h]
  __int64 v32; // [rsp+38h] [rbp-918h]
  size_t v33; // [rsp+48h] [rbp-908h] BYREF
  char *v34; // [rsp+50h] [rbp-900h] BYREF
  size_t v35; // [rsp+58h] [rbp-8F8h]
  _QWORD v36[2]; // [rsp+60h] [rbp-8F0h] BYREF
  const char *v37; // [rsp+70h] [rbp-8E0h] BYREF
  size_t v38; // [rsp+78h] [rbp-8D8h]
  _QWORD v39[2]; // [rsp+80h] [rbp-8D0h] BYREF
  _QWORD v40[2]; // [rsp+90h] [rbp-8C0h] BYREF
  _QWORD v41[2]; // [rsp+A0h] [rbp-8B0h] BYREF
  _QWORD v42[2]; // [rsp+B0h] [rbp-8A0h] BYREF
  _BYTE *v43; // [rsp+C0h] [rbp-890h]
  _BYTE *v44; // [rsp+C8h] [rbp-888h]
  int v45; // [rsp+D0h] [rbp-880h]
  unsigned __int64 *v46; // [rsp+D8h] [rbp-878h]
  void *v47; // [rsp+E0h] [rbp-870h] BYREF
  __int64 v48; // [rsp+E8h] [rbp-868h]
  __int64 v49; // [rsp+F0h] [rbp-860h]
  __int64 v50; // [rsp+F8h] [rbp-858h]
  int v51; // [rsp+100h] [rbp-850h]
  _QWORD *v52; // [rsp+108h] [rbp-848h]
  unsigned __int64 v53[2]; // [rsp+110h] [rbp-840h] BYREF
  _BYTE v54[2096]; // [rsp+120h] [rbp-830h] BYREF

  v53[0] = (unsigned __int64)v54;
  v46 = v53;
  v53[1] = 0x80000000000LL;
  v42[0] = &unk_49EFC48;
  v45 = 1;
  v44 = 0;
  v43 = 0;
  v42[1] = 0;
  sub_16E7A40((__int64)v42, 0, 0, 0);
  sub_16E7A90((__int64)v42, *((unsigned int *)a2 + 2));
  v4 = *a2;
  v32 = *a2 + 56LL * *((unsigned int *)a2 + 2);
  if ( *a2 != v32 )
  {
    do
    {
      v14 = *(const char **)v4;
      v34 = (char *)v36;
      if ( !v14 )
LABEL_51:
        sub_426248((__int64)"basic_string::_M_construct null not valid");
      v15 = (void *)strlen(v14);
      v47 = v15;
      v16 = (size_t)v15;
      if ( (unsigned __int64)v15 > 0xF )
      {
        v34 = (char *)sub_22409D0(&v34, &v47, 0);
        v21 = v34;
        v36[0] = v47;
      }
      else
      {
        if ( v15 == (void *)1 )
        {
          LOBYTE(v36[0]) = *v14;
          v5 = (char *)v36;
          goto LABEL_4;
        }
        if ( !v15 )
        {
          v5 = (char *)v36;
          goto LABEL_4;
        }
        v21 = (char *)v36;
      }
      memcpy(v21, v14, v16);
      v15 = v47;
      v5 = v34;
LABEL_4:
      v35 = (size_t)v15;
      *((_BYTE *)v15 + (_QWORD)v5) = 0;
      if ( !*(_DWORD *)(v4 + 48) )
        goto LABEL_5;
      if ( v35 == 0x3FFFFFFFFFFFFFFFLL )
        sub_4262D8((__int64)"basic_string::append");
      sub_2241490(&v34, ":", 1);
      v17 = *(unsigned int *)(v4 + 48);
      v40[1] = 0;
      v40[0] = v41;
      LOBYTE(v41[0]) = 0;
      v51 = 1;
      v50 = 0;
      v49 = 0;
      v48 = 0;
      v47 = &unk_49EFBE0;
      v52 = v40;
      sub_16E7A90((__int64)&v47, v17);
      if ( v50 != v48 )
        sub_16E7BA0((__int64 *)&v47);
      v37 = (const char *)v39;
      v18 = (_BYTE *)*v52;
      v19 = v52[1];
      if ( v19 + *v52 && !v18 )
        goto LABEL_51;
      v33 = v52[1];
      if ( v19 > 0xF )
      {
        n = v19;
        src = v18;
        v27 = (const char *)sub_22409D0(&v37, &v33, 0);
        v18 = src;
        v19 = n;
        v37 = v27;
        v28 = (char *)v27;
        v39[0] = v33;
      }
      else
      {
        if ( v19 == 1 )
        {
          LOBYTE(v39[0]) = *v18;
          v20 = (const char *)v39;
          goto LABEL_29;
        }
        if ( !v19 )
        {
          v20 = (const char *)v39;
          goto LABEL_29;
        }
        v28 = (char *)v39;
      }
      memcpy(v28, v18, v19);
      v19 = v33;
      v20 = v37;
LABEL_29:
      v38 = v19;
      v20[v19] = 0;
      sub_16E7BC0((__int64 *)&v47);
      if ( (_QWORD *)v40[0] != v41 )
        j_j___libc_free_0(v40[0], v41[0] + 1LL);
      sub_2241490(&v34, v37, v38);
      if ( v37 == (const char *)v39 )
      {
LABEL_5:
        v6 = v44;
        if ( v43 == v44 )
          goto LABEL_33;
        goto LABEL_6;
      }
      j_j___libc_free_0(v37, v39[0] + 1LL);
      v6 = v44;
      if ( v43 == v44 )
      {
LABEL_33:
        v7 = (_QWORD *)sub_16E7EE0((__int64)v42, " ", 1u);
        goto LABEL_7;
      }
LABEL_6:
      *v6 = 32;
      v7 = v42;
      ++v44;
LABEL_7:
      v8 = sub_16E7A90((__int64)v7, *(_QWORD *)(v4 + 40));
      v9 = *(_BYTE **)(v8 + 24);
      if ( *(_BYTE **)(v8 + 16) == v9 )
      {
        v8 = sub_16E7EE0(v8, " ", 1u);
      }
      else
      {
        *v9 = 32;
        ++*(_QWORD *)(v8 + 24);
      }
      v10 = sub_16E7A90(v8, *(_QWORD *)(v4 + 8));
      v11 = *(_BYTE **)(v10 + 24);
      if ( *(_BYTE **)(v10 + 16) == v11 )
      {
        v10 = sub_16E7EE0(v10, " ", 1u);
      }
      else
      {
        *v11 = 32;
        ++*(_QWORD *)(v10 + 24);
      }
      v12 = sub_16E7A90(v10, v35);
      v13 = *(_BYTE **)(v12 + 24);
      if ( *(_BYTE **)(v12 + 16) == v13 )
      {
        v12 = sub_16E7EE0(v12, " ", 1u);
      }
      else
      {
        *v13 = 32;
        ++*(_QWORD *)(v12 + 24);
      }
      sub_16E7EE0(v12, v34, v35);
      if ( v34 != (char *)v36 )
        j_j___libc_free_0(v34, v36[0] + 1LL);
      v4 += 56;
    }
    while ( v32 != v4 );
  }
  v22 = *((unsigned int *)v46 + 2);
  v23 = (const void *)*v46;
  v24 = a1 + 2;
  a1[1] = 0x4000000000LL;
  *a1 = a1 + 2;
  if ( (unsigned int)v22 > 0x40 )
  {
    sub_16CD150((__int64)a1, v24, v22, 1, v2, v3);
    v24 = (void *)(*a1 + *((unsigned int *)a1 + 2));
  }
  else
  {
    v25 = 0;
    if ( !v22 )
      goto LABEL_41;
  }
  memcpy(v24, v23, v22);
  v25 = v22 + *((_DWORD *)a1 + 2);
LABEL_41:
  *((_DWORD *)a1 + 2) = v25;
  v42[0] = &unk_49EFD28;
  sub_16E7960((__int64)v42);
  if ( (_BYTE *)v53[0] != v54 )
    _libc_free(v53[0]);
  return a1;
}
