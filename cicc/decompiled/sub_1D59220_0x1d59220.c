// Function: sub_1D59220
// Address: 0x1d59220
//
__int64 *__fastcall sub_1D59220(__int64 *a1, __int64 a2, __int64 *a3)
{
  __int64 v4; // rax
  __int64 v5; // rax
  int v6; // r8d
  int v7; // r9d
  __int64 v8; // rdx
  __int64 v9; // rbx
  _BYTE *v10; // rdx
  __int64 v11; // rax
  int v12; // edx
  unsigned int *v13; // rdx
  _BYTE *v14; // rdi
  __int64 v15; // r14
  __int64 v16; // r13
  char *v17; // rax
  _QWORD *v18; // rax
  void *v19; // rdi
  _BYTE *v20; // r14
  size_t v21; // r13
  _DWORD *v23; // rdx
  _DWORD *v24; // rdx
  __int64 v25; // rax
  _QWORD v27[2]; // [rsp+30h] [rbp-100h] BYREF
  _QWORD v28[2]; // [rsp+40h] [rbp-F0h] BYREF
  char *v29[2]; // [rsp+50h] [rbp-E0h] BYREF
  __int64 v30; // [rsp+60h] [rbp-D0h] BYREF
  void *v31; // [rsp+70h] [rbp-C0h] BYREF
  char *v32; // [rsp+78h] [rbp-B8h]
  __int64 v33; // [rsp+80h] [rbp-B0h]
  _DWORD *v34; // [rsp+88h] [rbp-A8h]
  int v35; // [rsp+90h] [rbp-A0h]
  _QWORD *v36; // [rsp+98h] [rbp-98h]
  _BYTE *v37; // [rsp+A0h] [rbp-90h] BYREF
  __int64 v38; // [rsp+A8h] [rbp-88h]
  _BYTE v39[32]; // [rsp+B0h] [rbp-80h] BYREF
  __int64 v40[4]; // [rsp+D0h] [rbp-60h] BYREF
  int v41; // [rsp+F0h] [rbp-40h]
  char **v42; // [rsp+F8h] [rbp-38h]

  v27[0] = v28;
  v31 = &unk_49EFBE0;
  v27[1] = 0;
  LOBYTE(v28[0]) = 0;
  v35 = 1;
  v34 = 0;
  v33 = 0;
  v32 = 0;
  v36 = v27;
  v4 = sub_16E7EE0((__int64)&v31, "SU(", 3u);
  v5 = sub_16E7A90(v4, *((unsigned int *)a3 + 48));
  v8 = *(_QWORD *)(v5 + 24);
  if ( (unsigned __int64)(*(_QWORD *)(v5 + 16) - v8) <= 2 )
  {
    sub_16E7EE0(v5, "): ", 3u);
    v9 = *a3;
    if ( v9 )
      goto LABEL_3;
  }
  else
  {
    *(_BYTE *)(v8 + 2) = 32;
    *(_WORD *)v8 = 14889;
    *(_QWORD *)(v5 + 24) += 3LL;
    v9 = *a3;
    if ( v9 )
    {
LABEL_3:
      v10 = v39;
      v37 = v39;
      v38 = 0x400000000LL;
      v11 = 0;
      while ( 1 )
      {
        *(_QWORD *)&v10[8 * v11] = v9;
        v12 = *(_DWORD *)(v9 + 56);
        v11 = (unsigned int)(v38 + 1);
        LODWORD(v38) = v38 + 1;
        if ( !v12 )
          break;
        v13 = (unsigned int *)(*(_QWORD *)(v9 + 32) + 40LL * (unsigned int)(v12 - 1));
        v9 = *(_QWORD *)v13;
        if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)v13 + 40LL) + 16LL * v13[2]) != 111 )
          break;
        if ( HIDWORD(v38) <= (unsigned int)v11 )
        {
          sub_16CD150((__int64)&v37, v39, 0, 8, v6, v7);
          v11 = (unsigned int)v38;
        }
        v10 = v37;
      }
      while ( 1 )
      {
        v14 = v37;
        if ( !(_DWORD)v11 )
          break;
        v15 = *(_QWORD *)&v37[8 * v11 - 8];
        v16 = *(_QWORD *)(a2 + 624);
        sub_2095B00(v29, v15, v16);
        v41 = 1;
        v40[0] = (__int64)&unk_49EFBE0;
        memset(&v40[1], 0, 24);
        v42 = v29;
        sub_20945B0(v15, v40, v16);
        sub_16E7BC0(v40);
        sub_16E7EE0((__int64)&v31, v29[0], (size_t)v29[1]);
        if ( (__int64 *)v29[0] != &v30 )
          j_j___libc_free_0(v29[0], v30 + 1);
        LODWORD(v38) = v38 - 1;
        if ( !(_DWORD)v38 )
        {
          v14 = v37;
          break;
        }
        v23 = v34;
        if ( (unsigned __int64)(v33 - (_QWORD)v34) <= 4 )
        {
          sub_16E7EE0((__int64)&v31, "\n    ", 5u);
          v11 = (unsigned int)v38;
        }
        else
        {
          *v34 = 538976266;
          *((_BYTE *)v23 + 4) = 32;
          v11 = (unsigned int)v38;
          v34 = (_DWORD *)((char *)v34 + 5);
        }
      }
      if ( v14 != v39 )
        _libc_free((unsigned __int64)v14);
      goto LABEL_13;
    }
  }
  v24 = v34;
  if ( (unsigned __int64)(v33 - (_QWORD)v34) <= 0xC )
  {
    sub_16E7EE0((__int64)&v31, "CROSS RC COPY", 0xDu);
LABEL_13:
    v17 = (char *)v34;
    goto LABEL_14;
  }
  v34[2] = 1347371808;
  *(_QWORD *)v24 = 0x43522053534F5243LL;
  *((_BYTE *)v24 + 12) = 89;
  v17 = (char *)v34 + 13;
  v34 = (_DWORD *)((char *)v34 + 13);
LABEL_14:
  if ( v32 != v17 )
    sub_16E7BA0((__int64 *)&v31);
  v18 = v36;
  v19 = a1 + 2;
  *a1 = (__int64)(a1 + 2);
  v20 = (_BYTE *)*v18;
  v21 = v18[1];
  if ( v21 + *v18 && !v20 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v40[0] = v18[1];
  if ( v21 > 0xF )
  {
    v25 = sub_22409D0(a1, v40, 0);
    *a1 = v25;
    v19 = (void *)v25;
    a1[2] = v40[0];
LABEL_35:
    memcpy(v19, v20, v21);
    v21 = v40[0];
    v19 = (void *)*a1;
    goto LABEL_21;
  }
  if ( v21 == 1 )
  {
    *((_BYTE *)a1 + 16) = *v20;
    goto LABEL_21;
  }
  if ( v21 )
    goto LABEL_35;
LABEL_21:
  a1[1] = v21;
  *((_BYTE *)v19 + v21) = 0;
  sub_16E7BC0((__int64 *)&v31);
  if ( (_QWORD *)v27[0] != v28 )
    j_j___libc_free_0(v27[0], v28[0] + 1LL);
  return a1;
}
