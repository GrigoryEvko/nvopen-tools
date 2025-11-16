// Function: sub_395A2F0
// Address: 0x395a2f0
//
__int64 __fastcall sub_395A2F0(__int64 a1, __int64 a2, __int64 a3, _DWORD *a4, _DWORD *a5)
{
  __int64 v8; // rax
  int v9; // r9d
  __int64 *v10; // r15
  int v11; // r10d
  int v12; // eax
  __int64 v13; // r14
  __int64 *v14; // rax
  unsigned int v15; // edx
  unsigned int v16; // eax
  __int64 v17; // r14
  int v18; // eax
  _QWORD *v19; // rdi
  int v21; // esi
  __int64 v22; // rdx
  __int64 v23; // rax
  __int64 v24; // rdx
  int v25; // esi
  __int64 v26; // rax
  int v27; // [rsp+8h] [rbp-98h]
  int v28; // [rsp+8h] [rbp-98h]
  int v29; // [rsp+Ch] [rbp-94h]
  int v30; // [rsp+Ch] [rbp-94h]
  int v31; // [rsp+10h] [rbp-90h]
  __int64 v32; // [rsp+28h] [rbp-78h] BYREF
  _QWORD v33[2]; // [rsp+30h] [rbp-70h] BYREF
  char v34; // [rsp+40h] [rbp-60h]
  char v35; // [rsp+41h] [rbp-5Fh]
  _QWORD *v36; // [rsp+50h] [rbp-50h] BYREF
  __int64 v37; // [rsp+58h] [rbp-48h]
  _QWORD v38[8]; // [rsp+60h] [rbp-40h] BYREF

  v36 = v38;
  v37 = 0x200000000LL;
  if ( *(_BYTE *)(*(_QWORD *)a4 + 16LL) > 0x17u )
  {
    v38[0] = *(_QWORD *)a4;
    LODWORD(v37) = 1;
  }
  if ( *(_BYTE *)(*(_QWORD *)a5 + 16LL) > 0x17u )
  {
    v38[(unsigned int)v37] = *(_QWORD *)a5;
    LODWORD(v37) = v37 + 1;
  }
  v8 = sub_3958EE0(a2, (__int64 *)&v36, a3);
  v9 = a5[3];
  v10 = *(__int64 **)a4;
  v11 = 11;
  v32 = v8;
  v12 = a4[3];
  v13 = *(_QWORD *)a5;
  if ( v12 != v9 )
  {
    v9 = 11;
    v11 = 13;
    if ( v12 != 11 )
    {
      v14 = v10;
      v11 = 13;
      v10 = *(__int64 **)a5;
      v9 = 11;
      v13 = (__int64)v14;
    }
  }
  v15 = *(_DWORD *)(*v10 + 8) >> 8;
  v16 = *(_DWORD *)(*(_QWORD *)v13 + 8LL) >> 8;
  if ( v15 > v16 )
  {
    v24 = *v10;
    v35 = 1;
    v28 = v11;
    v33[0] = "tree.ext";
    v30 = v9;
    v25 = a5[2];
    v34 = 3;
    v26 = sub_3958FF0(v13, v25, v24, &v32, (__int64)v33);
    v9 = v30;
    v11 = v28;
    v13 = v26;
  }
  else if ( v15 < v16 )
  {
    v21 = a4[2];
    v22 = *(_QWORD *)v13;
    v27 = v11;
    v29 = v9;
    v35 = 1;
    v33[0] = "tree.ext";
    v34 = 3;
    v23 = sub_3958FF0((__int64)v10, v21, v22, &v32, (__int64)v33);
    v11 = v27;
    v9 = v29;
    v10 = (__int64 *)v23;
  }
  v31 = v9;
  v35 = 1;
  v33[0] = "tree.add";
  v34 = 3;
  v17 = sub_15FB440(v11, v10, v13, (__int64)v33, 0);
  sub_15F2180(v17, v32);
  v18 = 1;
  if ( a4[2] == 2 )
  {
    v18 = a5[2];
    if ( v18 != 2 )
      v18 = 1;
  }
  *(_QWORD *)a1 = v17;
  v19 = v36;
  *(_DWORD *)(a1 + 8) = v18;
  *(_DWORD *)(a1 + 12) = v31;
  *(_QWORD *)(a1 + 16) = v17;
  if ( v19 != v38 )
    _libc_free((unsigned __int64)v19);
  return a1;
}
