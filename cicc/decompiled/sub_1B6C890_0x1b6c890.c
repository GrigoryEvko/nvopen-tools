// Function: sub_1B6C890
// Address: 0x1b6c890
//
__int64 __fastcall sub_1B6C890(__int64 a1, unsigned __int64 *a2, __int64 a3, __int64 a4, __int64 a5)
{
  char *v6; // r12
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // rbx
  _QWORD *v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  const char *v16; // rax
  unsigned int v17; // r12d
  __int64 v18; // rax
  __int64 *v19; // rdi
  const char *v21; // rax
  __int64 v22; // [rsp+8h] [rbp-98h]
  const char *v24; // [rsp+20h] [rbp-80h] BYREF
  char v25; // [rsp+30h] [rbp-70h]
  char v26; // [rsp+31h] [rbp-6Fh]
  unsigned __int64 v27[2]; // [rsp+40h] [rbp-60h] BYREF
  _BYTE v28[80]; // [rsp+50h] [rbp-50h] BYREF

  v6 = (char *)a2;
  v27[0] = (unsigned __int64)v28;
  v27[1] = 0x2000000000LL;
  v7 = sub_16FD110(a3, (unsigned __int64)a2, a3, a4, a5);
  if ( *(_DWORD *)(v7 + 32) != 1 )
  {
    v26 = 1;
    v21 = "rewrite type must be a scalar";
LABEL_15:
    v24 = v21;
    v25 = 3;
    v18 = sub_16FD110(a3, (unsigned __int64)a2, v8, v9, v10);
    goto LABEL_10;
  }
  v11 = v7;
  v12 = sub_16FD200(a3, (unsigned __int64)a2, v8, v9, v10);
  if ( *((_DWORD *)v12 + 8) != 4 )
  {
    v26 = 1;
    v24 = "rewrite descriptor must be a map";
    v25 = 3;
    v18 = (__int64)sub_16FD200(a3, (unsigned __int64)a2, v13, v14, v15);
LABEL_10:
    v19 = (__int64 *)v6;
    v17 = 0;
    sub_16F8270(v19, v18, (__int64)&v24);
    goto LABEL_11;
  }
  v22 = (__int64)v12;
  a2 = v27;
  v16 = sub_16F8F10(v11, v27);
  v9 = v22;
  if ( v8 == 8 )
  {
    v8 = 0x6E6F6974636E7566LL;
    if ( *(_QWORD *)v16 == 0x6E6F6974636E7566LL )
    {
      v17 = sub_1B699A0(a1, v6, v11, v22, a4);
      goto LABEL_11;
    }
LABEL_17:
    v26 = 1;
    v21 = "unknown rewrite type";
    goto LABEL_15;
  }
  if ( v8 != 15 )
  {
    if ( v8 == 12 )
    {
      v8 = 0x61206C61626F6C67LL;
      if ( *(_QWORD *)v16 == 0x61206C61626F6C67LL && *((_DWORD *)v16 + 2) == 1935763820 )
      {
        v17 = sub_1B6BAB0(a1, (unsigned __int64 *)v6, v11, v22, a4);
        goto LABEL_11;
      }
    }
    goto LABEL_17;
  }
  v8 = 0x76206C61626F6C67LL;
  if ( *(_QWORD *)v16 != 0x76206C61626F6C67LL
    || *((_DWORD *)v16 + 2) != 1634300513
    || *((_WORD *)v16 + 6) != 27746
    || v16[14] != 101 )
  {
    goto LABEL_17;
  }
  v17 = sub_1B6ACD0(a1, (unsigned __int64 *)v6, v11, v22, a4);
LABEL_11:
  if ( (_BYTE *)v27[0] != v28 )
    _libc_free(v27[0]);
  return v17;
}
