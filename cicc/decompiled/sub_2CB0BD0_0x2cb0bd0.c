// Function: sub_2CB0BD0
// Address: 0x2cb0bd0
//
__int64 __fastcall sub_2CB0BD0(__int64 a1, __int64 a2, _BYTE *a3, __int64 a4, __int64 a5)
{
  _BYTE *v8; // rax
  int v9; // ecx
  __int64 v10; // r15
  int v11; // r10d
  int v12; // eax
  __int64 v13; // r14
  __int64 v14; // rax
  unsigned int v15; // edx
  unsigned int v16; // eax
  __int64 v17; // r14
  int v18; // eax
  _QWORD *v19; // rdi
  int v21; // r9d
  __int64 v22; // rdx
  __int64 v23; // rax
  int v24; // esi
  __int64 v25; // rdx
  __int64 v26; // rax
  int v27; // [rsp+18h] [rbp-98h]
  int v28; // [rsp+18h] [rbp-98h]
  int v29; // [rsp+1Ch] [rbp-94h]
  _BYTE *v30; // [rsp+28h] [rbp-88h] BYREF
  _QWORD *v31; // [rsp+30h] [rbp-80h] BYREF
  __int64 v32; // [rsp+38h] [rbp-78h]
  _QWORD v33[2]; // [rsp+40h] [rbp-70h] BYREF
  _QWORD v34[4]; // [rsp+50h] [rbp-60h] BYREF
  char v35; // [rsp+70h] [rbp-40h]
  char v36; // [rsp+71h] [rbp-3Fh]

  v31 = v33;
  v32 = 0x200000000LL;
  if ( **(_BYTE **)a4 > 0x1Cu )
  {
    v33[0] = *(_QWORD *)a4;
    LODWORD(v32) = 1;
  }
  if ( **(_BYTE **)a5 > 0x1Cu )
  {
    v33[(unsigned int)v32] = *(_QWORD *)a5;
    LODWORD(v32) = v32 + 1;
  }
  v8 = sub_2CAF220(a2, (__int64 *)&v31, a3);
  v9 = *(_DWORD *)(a5 + 12);
  v10 = *(_QWORD *)a4;
  v11 = 13;
  v30 = v8;
  v12 = *(_DWORD *)(a4 + 12);
  v29 = v9;
  v13 = *(_QWORD *)a5;
  if ( v12 != v9 )
  {
    v29 = 13;
    if ( v12 == 13 )
    {
      v11 = 15;
    }
    else
    {
      v14 = v10;
      v11 = 15;
      v10 = *(_QWORD *)a5;
      v13 = v14;
    }
  }
  v15 = *(_DWORD *)(*(_QWORD *)(v10 + 8) + 8LL) >> 8;
  v16 = *(_DWORD *)(*(_QWORD *)(v13 + 8) + 8LL) >> 8;
  if ( v15 > v16 )
  {
    v24 = *(_DWORD *)(a5 + 8);
    v25 = *(_QWORD *)(v10 + 8);
    v28 = v11;
    v36 = 1;
    v34[0] = "tree.ext";
    v35 = 3;
    v26 = sub_2CAF330(v13, v24, v25, (__int64 *)&v30, (__int64)v34);
    v11 = v28;
    v13 = v26;
  }
  else if ( v15 < v16 )
  {
    v21 = *(_DWORD *)(a4 + 8);
    v22 = *(_QWORD *)(v13 + 8);
    v36 = 1;
    v27 = v11;
    v34[0] = "tree.ext";
    v35 = 3;
    v23 = sub_2CAF330(v10, v21, v22, (__int64 *)&v30, (__int64)v34);
    v11 = v27;
    v10 = v23;
  }
  v36 = 1;
  v34[0] = "tree.add";
  v35 = 3;
  v17 = sub_B504D0(v11, v10, v13, (__int64)v34, 0, 0);
  sub_B43DD0(v17, (__int64)v30);
  v18 = 1;
  if ( *(_DWORD *)(a4 + 8) == 2 )
  {
    v18 = *(_DWORD *)(a5 + 8);
    if ( v18 != 2 )
      v18 = 1;
  }
  *(_DWORD *)(a1 + 8) = v18;
  *(_QWORD *)a1 = v17;
  v19 = v31;
  *(_DWORD *)(a1 + 12) = v29;
  *(_QWORD *)(a1 + 16) = v17;
  if ( v19 != v33 )
    _libc_free((unsigned __int64)v19);
  return a1;
}
