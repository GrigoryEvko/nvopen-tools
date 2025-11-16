// Function: sub_D01830
// Address: 0xd01830
//
__int64 __fastcall sub_D01830(__int64 a1, _DWORD *a2, _DWORD *a3)
{
  int v5; // eax
  unsigned int v6; // r12d
  int v7; // edx
  int v8; // eax
  unsigned __int64 v9; // r14
  int v10; // eax
  __int64 v11; // rax
  unsigned int v13; // r12d
  unsigned __int64 v14; // rdx
  int v15; // r15d
  __int64 v16; // rax
  bool v17; // cc
  __int64 v18; // rdi
  unsigned int v19; // r12d
  unsigned __int64 v20; // rdx
  int v21; // r15d
  __int64 v22; // rax
  __int64 v23; // rdi
  unsigned int v24; // r14d
  int v25; // r15d
  __int64 v26; // rax
  __int64 v27; // rdi
  __int64 v28; // [rsp+0h] [rbp-70h]
  __int64 v29; // [rsp+0h] [rbp-70h]
  unsigned __int64 v30; // [rsp+8h] [rbp-68h]
  __int64 v31; // [rsp+8h] [rbp-68h]
  unsigned __int64 v32; // [rsp+8h] [rbp-68h]
  __int64 v33; // [rsp+8h] [rbp-68h]
  __int64 v34; // [rsp+8h] [rbp-68h]
  __int64 v35; // [rsp+8h] [rbp-68h]
  __int64 v36; // [rsp+8h] [rbp-68h]
  unsigned __int64 v37; // [rsp+10h] [rbp-60h] BYREF
  unsigned int v38; // [rsp+18h] [rbp-58h]
  unsigned __int64 v39; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v40; // [rsp+28h] [rbp-48h]
  __int64 v41; // [rsp+30h] [rbp-40h] BYREF
  int v42; // [rsp+38h] [rbp-38h]

  v5 = a2[4];
  v6 = a3[2];
  if ( !v5 )
  {
    v7 = a2[3];
    if ( !v7 )
      goto LABEL_3;
LABEL_13:
    v19 = v7 + v6;
    sub_C44830((__int64)&v41, a3 + 4, v19);
    sub_C44830((__int64)&v39, a3, v19);
    v6 = v40;
    v20 = v39;
    v21 = v42;
    v22 = v41;
    if ( a3[2] > 0x40u && *(_QWORD *)a3 )
    {
      v29 = v41;
      v32 = v39;
      j_j___libc_free_0_0(*(_QWORD *)a3);
      v22 = v29;
      v20 = v32;
    }
    v17 = a3[6] <= 0x40u;
    *(_QWORD *)a3 = v20;
    a3[2] = v6;
    if ( !v17 )
    {
      v23 = *((_QWORD *)a3 + 2);
      if ( v23 )
      {
        v33 = v22;
        j_j___libc_free_0_0(v23);
        v6 = a3[2];
        v22 = v33;
      }
    }
    *((_QWORD *)a3 + 2) = v22;
    a3[6] = v21;
    v8 = a2[2];
    if ( !v8 )
      goto LABEL_4;
    goto LABEL_20;
  }
  v13 = v6 - v5;
  sub_C44740((__int64)&v41, (char **)a3 + 2, v13);
  sub_C44740((__int64)&v39, (char **)a3, v13);
  v6 = v40;
  v14 = v39;
  v15 = v42;
  v16 = v41;
  if ( a3[2] > 0x40u && *(_QWORD *)a3 )
  {
    v28 = v41;
    v30 = v39;
    j_j___libc_free_0_0(*(_QWORD *)a3);
    v16 = v28;
    v14 = v30;
  }
  v17 = a3[6] <= 0x40u;
  *(_QWORD *)a3 = v14;
  a3[2] = v6;
  if ( !v17 )
  {
    v18 = *((_QWORD *)a3 + 2);
    if ( v18 )
    {
      v31 = v16;
      j_j___libc_free_0_0(v18);
      v6 = a3[2];
      v16 = v31;
    }
  }
  *((_QWORD *)a3 + 2) = v16;
  a3[6] = v15;
  v7 = a2[3];
  if ( v7 )
    goto LABEL_13;
LABEL_3:
  v8 = a2[2];
  if ( !v8 )
  {
LABEL_4:
    v9 = *(_QWORD *)a3;
    goto LABEL_5;
  }
LABEL_20:
  v24 = v8 + v6;
  sub_C449B0((__int64)&v37, (const void **)a3, v8 + v6);
  if ( v38 != v6 )
  {
    if ( v6 > 0x3F || v38 > 0x40 )
      sub_C43C90(&v37, v6, v38);
    else
      v37 |= 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v6 - (unsigned __int8)v38 + 64) << v6;
  }
  sub_C449B0((__int64)&v41, (const void **)a3 + 2, v24);
  v6 = v38;
  v40 = v38;
  if ( v38 > 0x40 )
  {
    sub_C43780((__int64)&v39, (const void **)&v37);
    v6 = v40;
    v9 = v39;
    v25 = v42;
    v26 = v41;
    if ( v38 > 0x40 && v37 )
    {
      v36 = v41;
      j_j___libc_free_0_0(v37);
      v26 = v36;
    }
  }
  else
  {
    v9 = v37;
    v25 = v42;
    v26 = v41;
  }
  if ( a3[2] > 0x40u && *(_QWORD *)a3 )
  {
    v34 = v26;
    j_j___libc_free_0_0(*(_QWORD *)a3);
    v26 = v34;
  }
  v17 = a3[6] <= 0x40u;
  *(_QWORD *)a3 = v9;
  a3[2] = v6;
  if ( !v17 )
  {
    v27 = *((_QWORD *)a3 + 2);
    if ( v27 )
    {
      v35 = v26;
      j_j___libc_free_0_0(v27);
      v6 = a3[2];
      v9 = *(_QWORD *)a3;
      v26 = v35;
    }
  }
  *((_QWORD *)a3 + 2) = v26;
  a3[6] = v25;
LABEL_5:
  *(_DWORD *)(a1 + 8) = v6;
  *(_QWORD *)a1 = v9;
  v10 = a3[6];
  a3[2] = 0;
  *(_DWORD *)(a1 + 24) = v10;
  v11 = *((_QWORD *)a3 + 2);
  a3[6] = 0;
  *(_QWORD *)(a1 + 16) = v11;
  return a1;
}
