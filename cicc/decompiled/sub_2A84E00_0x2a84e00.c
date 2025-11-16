// Function: sub_2A84E00
// Address: 0x2a84e00
//
__int64 __fastcall sub_2A84E00(__int64 a1, unsigned __int64 *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 ***v6; // r12
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // rbx
  _QWORD *v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  char *v16; // rax
  unsigned int v17; // r12d
  __int64 v18; // rax
  const char *v20; // rax
  __int64 v21; // [rsp+8h] [rbp-B8h]
  const char *v23; // [rsp+20h] [rbp-A0h] BYREF
  char v24; // [rsp+40h] [rbp-80h]
  char v25; // [rsp+41h] [rbp-7Fh]
  unsigned __int64 v26[3]; // [rsp+50h] [rbp-70h] BYREF
  _BYTE v27[88]; // [rsp+68h] [rbp-58h] BYREF

  v6 = (__int64 ***)a2;
  v26[0] = (unsigned __int64)v27;
  v26[1] = 0;
  v26[2] = 32;
  v7 = sub_CAE820(a3, (unsigned __int64)a2, a3, a4, a5);
  if ( *(_DWORD *)(v7 + 32) != 1 )
  {
    v25 = 1;
    v20 = "rewrite type must be a scalar";
LABEL_16:
    v23 = v20;
    v24 = 3;
    v18 = sub_CAE820(a3, (unsigned __int64)a2, v8, v9, v10);
    goto LABEL_11;
  }
  v11 = v7;
  v12 = sub_CAE940(a3, (unsigned __int64)a2, v8, v9, v10);
  if ( *((_DWORD *)v12 + 8) != 4 )
  {
    v25 = 1;
    v23 = "rewrite descriptor must be a map";
    v24 = 3;
    v18 = (__int64)sub_CAE940(a3, (unsigned __int64)a2, v13, v14, v15);
LABEL_11:
    sub_CA89D0(v6, v18, (__int64)&v23, 0);
    v17 = 0;
    goto LABEL_12;
  }
  v21 = (__int64)v12;
  a2 = v26;
  v16 = sub_CA8C30(v11, v26);
  v9 = v21;
  if ( v8 == 8 )
  {
    v8 = 0x6E6F6974636E7566LL;
    if ( *(_QWORD *)v16 == 0x6E6F6974636E7566LL )
    {
      v17 = sub_2A81E50(a1, (unsigned __int64 *)v6, v11, v21, a4);
      goto LABEL_12;
    }
LABEL_18:
    v25 = 1;
    v20 = "unknown rewrite type";
    goto LABEL_16;
  }
  if ( v8 == 15 )
  {
    v8 = 0x76206C61626F6C67LL;
    if ( *(_QWORD *)v16 == 0x76206C61626F6C67LL
      && *((_DWORD *)v16 + 2) == 1634300513
      && *((_WORD *)v16 + 6) == 27746
      && v16[14] == 101 )
    {
      v17 = sub_2A83180(a1, (unsigned __int64 *)v6, v11, v21, a4);
      goto LABEL_12;
    }
    goto LABEL_18;
  }
  if ( v8 != 12 )
    goto LABEL_18;
  v8 = 0x61206C61626F6C67LL;
  if ( *(_QWORD *)v16 != 0x61206C61626F6C67LL || *((_DWORD *)v16 + 2) != 1935763820 )
    goto LABEL_18;
  v17 = sub_2A83FC0(a1, (unsigned __int64 *)v6, v11, v21, a4);
LABEL_12:
  if ( (_BYTE *)v26[0] != v27 )
    _libc_free(v26[0]);
  return v17;
}
