// Function: sub_1D49610
// Address: 0x1d49610
//
char __fastcall sub_1D49610(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rax
  char v9; // di
  __int64 v10; // rax
  __int64 v11; // r14
  unsigned int v12; // eax
  unsigned __int64 v13; // r8
  unsigned int v14; // r15d
  __int64 *v15; // r14
  char result; // al
  __int64 v17; // rdx
  unsigned __int64 v18; // rbx
  unsigned __int64 v19; // rdx
  __int64 v20; // rbx
  __int64 v21; // rdi
  __int64 v22; // rdi
  char v24; // [rsp+8h] [rbp-88h]
  char v25; // [rsp+8h] [rbp-88h]
  char v26; // [rsp+8h] [rbp-88h]
  char v27; // [rsp+8h] [rbp-88h]
  unsigned __int64 v28; // [rsp+10h] [rbp-80h] BYREF
  unsigned int v29; // [rsp+18h] [rbp-78h]
  __int64 v30; // [rsp+20h] [rbp-70h] BYREF
  int v31; // [rsp+28h] [rbp-68h]
  __int64 v32; // [rsp+30h] [rbp-60h] BYREF
  unsigned int v33; // [rsp+38h] [rbp-58h]
  __int64 v34; // [rsp+40h] [rbp-50h] BYREF
  __int64 v35; // [rsp+48h] [rbp-48h]
  __int64 v36; // [rsp+50h] [rbp-40h] BYREF
  __int64 v37; // [rsp+58h] [rbp-38h]

  v8 = *(_QWORD *)(a2 + 40) + 16LL * (unsigned int)a3;
  v9 = *(_BYTE *)v8;
  v10 = *(_QWORD *)(v8 + 8);
  v11 = *(_QWORD *)(a4 + 88);
  LOBYTE(v34) = v9;
  v35 = v10;
  if ( v9 )
    v12 = sub_1D46910(v9);
  else
    v12 = sub_1F58D40(&v34, a2, a3, a4, a5, a6);
  v29 = v12;
  if ( v12 > 0x40 )
  {
    sub_16A4EF0((__int64)&v28, a5, 0);
    v14 = *(_DWORD *)(v11 + 32);
    if ( v14 > 0x40 )
      goto LABEL_5;
LABEL_12:
    v17 = *(_QWORD *)(v11 + 24);
    v18 = v28;
    result = 1;
    if ( v17 == v28 )
      goto LABEL_7;
    if ( (v17 & ~v28) != 0 )
    {
      result = 0;
      goto LABEL_7;
    }
    goto LABEL_17;
  }
  v13 = a5 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v12);
  v14 = *(_DWORD *)(v11 + 32);
  v28 = v13;
  if ( v14 <= 0x40 )
    goto LABEL_12;
LABEL_5:
  v15 = (__int64 *)(v11 + 24);
  result = sub_16A5220((__int64)v15, (const void **)&v28);
  if ( result )
    goto LABEL_7;
  result = sub_16A5A00(v15, (__int64 *)&v28);
  if ( !result )
    goto LABEL_7;
  v33 = v14;
  sub_16A4FD0((__int64)&v32, (const void **)v15);
  v14 = v33;
  if ( v33 <= 0x40 )
  {
    v17 = v32;
    v18 = v28;
LABEL_17:
    v19 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v14) & ~v17;
LABEL_18:
    v20 = v19 & v18;
    v34 = 0;
    v21 = *(_QWORD *)(a1 + 272);
    v35 = 1;
    v36 = 0;
    v37 = 1;
    sub_1D1F820(v21, a2, a3, (unsigned __int64 *)&v34, 0);
LABEL_19:
    result = (v20 & ~v36) == 0;
    goto LABEL_20;
  }
  sub_16A8F40(&v32);
  v14 = v33;
  v19 = v32;
  v33 = 0;
  LODWORD(v35) = v14;
  v34 = v32;
  if ( v14 <= 0x40 )
  {
    v18 = v28;
    goto LABEL_18;
  }
  sub_16A8890(&v34, (__int64 *)&v28);
  v14 = v35;
  v20 = v34;
  v31 = v35;
  v30 = v34;
  if ( v33 > 0x40 && v32 )
    j_j___libc_free_0_0(v32);
  v34 = 0;
  v22 = *(_QWORD *)(a1 + 272);
  v35 = 1;
  v36 = 0;
  v37 = 1;
  sub_1D1F820(v22, a2, a3, (unsigned __int64 *)&v34, 0);
  if ( v14 <= 0x40 )
    goto LABEL_19;
  result = sub_16A5A00(&v30, &v36);
LABEL_20:
  if ( (unsigned int)v37 > 0x40 && v36 )
  {
    v25 = result;
    j_j___libc_free_0_0(v36);
    result = v25;
  }
  if ( (unsigned int)v35 > 0x40 && v34 )
  {
    v26 = result;
    j_j___libc_free_0_0(v34);
    result = v26;
  }
  if ( v14 > 0x40 && v20 )
  {
    v27 = result;
    j_j___libc_free_0_0(v20);
    result = v27;
  }
LABEL_7:
  if ( v29 > 0x40 )
  {
    if ( v28 )
    {
      v24 = result;
      j_j___libc_free_0_0(v28);
      return v24;
    }
  }
  return result;
}
