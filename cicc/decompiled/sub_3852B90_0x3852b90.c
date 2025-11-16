// Function: sub_3852B90
// Address: 0x3852b90
//
__int64 __fastcall sub_3852B90(__int64 a1, unsigned __int8 *a2, double a3, double a4, double a5)
{
  int v7; // eax
  __int64 v8; // rbx
  __int64 v9; // r15
  int v10; // esi
  __int64 v11; // rcx
  unsigned int v12; // edx
  __int64 *v13; // rax
  __int64 v14; // rdi
  __int64 v15; // r14
  unsigned int v16; // edx
  unsigned __int64 v17; // rax
  int v18; // eax
  int v19; // edx
  __int64 v20; // rcx
  unsigned int v21; // eax
  __int64 *v22; // rsi
  __int64 v23; // rdi
  __int64 v24; // rdx
  void *v25; // rax
  unsigned int v26; // r14d
  int v28; // eax
  __int64 *v29; // rax
  __int64 *v30; // rbx
  __int64 *v31; // rax
  __int64 v32; // rax
  __int64 v33; // rbx
  int v34; // r8d
  int v35; // esi
  int v36; // r8d
  unsigned __int64 v37; // [rsp+0h] [rbp-70h] BYREF
  unsigned int v38; // [rsp+8h] [rbp-68h]
  unsigned __int64 v39; // [rsp+10h] [rbp-60h] BYREF
  unsigned int v40; // [rsp+18h] [rbp-58h]
  __int64 v41; // [rsp+20h] [rbp-50h] BYREF
  unsigned __int64 v42; // [rsp+28h] [rbp-48h] BYREF
  unsigned int v43; // [rsp+30h] [rbp-40h]

  v7 = *(_DWORD *)(a1 + 256);
  v8 = *((_QWORD *)a2 - 6);
  v38 = 1;
  v37 = 0;
  v9 = *((_QWORD *)a2 - 3);
  v40 = 1;
  v39 = 0;
  if ( !v7 )
  {
LABEL_23:
    v37 = 0;
    v38 = 1;
    goto LABEL_10;
  }
  v10 = v7 - 1;
  v11 = *(_QWORD *)(a1 + 240);
  v12 = (v7 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
  v13 = (__int64 *)(v11 + 32LL * v12);
  v14 = *v13;
  if ( v8 != *v13 )
  {
    v28 = 1;
    while ( v14 != -8 )
    {
      v34 = v28 + 1;
      v12 = v10 & (v28 + v12);
      v13 = (__int64 *)(v11 + 32LL * v12);
      v14 = *v13;
      if ( v8 == *v13 )
        goto LABEL_3;
      v28 = v34;
    }
    goto LABEL_23;
  }
LABEL_3:
  v15 = v13[1];
  v41 = v15;
  v16 = *((_DWORD *)v13 + 6);
  v43 = v16;
  if ( v16 > 0x40 )
  {
    sub_16A4FD0((__int64)&v42, (const void **)v13 + 2);
    v15 = v41;
    v37 = v42;
    v38 = v43;
    if ( !v41 )
      goto LABEL_10;
    v18 = *(_DWORD *)(a1 + 256);
    if ( v18 )
      goto LABEL_6;
LABEL_24:
    v41 = 0;
    v43 = 1;
    v42 = 0;
    v39 = 0;
    v40 = 1;
    goto LABEL_10;
  }
  v17 = v13[2];
  v38 = v16;
  v37 = v17;
  if ( !v15 )
    goto LABEL_10;
  v18 = *(_DWORD *)(a1 + 256);
  if ( !v18 )
    goto LABEL_24;
LABEL_6:
  v19 = v18 - 1;
  v20 = *(_QWORD *)(a1 + 240);
  v21 = (v18 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
  v22 = (__int64 *)(v20 + 32LL * v21);
  v23 = *v22;
  if ( v9 != *v22 )
  {
    v35 = 1;
    while ( v23 != -8 )
    {
      v36 = v35 + 1;
      v21 = v19 & (v35 + v21);
      v22 = (__int64 *)(v20 + 32LL * v21);
      v23 = *v22;
      if ( v9 == *v22 )
        goto LABEL_7;
      v35 = v36;
    }
    goto LABEL_24;
  }
LABEL_7:
  v24 = v22[1];
  v41 = v24;
  v43 = *((_DWORD *)v22 + 6);
  if ( v43 > 0x40 )
  {
    v25 = sub_16A4FD0((__int64)&v42, (const void **)v22 + 2);
    v24 = v41;
  }
  else
  {
    v25 = (void *)v22[2];
    v42 = (unsigned __int64)v25;
  }
  LOBYTE(v20) = v24 != 0;
  LOBYTE(v25) = v15 == v24;
  v26 = (unsigned int)v25 & v20;
  v39 = v42;
  v40 = v43;
  if ( ((unsigned __int8)v25 & (v24 != 0)) != 0 )
  {
    v29 = (__int64 *)sub_16498A0(v8);
    v30 = (__int64 *)sub_159C0E0(v29, (__int64)&v37);
    v31 = (__int64 *)sub_16498A0(v9);
    v32 = sub_159C0E0(v31, (__int64)&v39);
    v33 = sub_15A2B60(v30, v32, 0, 0, a3, a4, a5);
    if ( v33 )
    {
      v41 = (__int64)a2;
      sub_38526A0(a1 + 136, &v41)[1] = v33;
      ++*(_DWORD *)(a1 + 548);
      goto LABEL_11;
    }
  }
LABEL_10:
  v26 = sub_38528E0(a1, a2);
LABEL_11:
  if ( v40 > 0x40 && v39 )
    j_j___libc_free_0_0(v39);
  if ( v38 > 0x40 && v37 )
    j_j___libc_free_0_0(v37);
  return v26;
}
