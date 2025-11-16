// Function: sub_35BEB00
// Address: 0x35beb00
//
__int64 __fastcall sub_35BEB00(_QWORD *a1, int *a2)
{
  int v3; // edx
  volatile signed __int32 *v4; // rax
  volatile signed __int32 *v5; // rcx
  __int64 v6; // rdi
  __int64 v7; // rdx
  __int64 v8; // rax
  unsigned int v9; // r13d
  __int64 v10; // rbx
  volatile signed __int32 *v11; // rdi
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // rdi
  __int64 v14; // rdx
  volatile signed __int32 *v15; // rax
  volatile signed __int32 *v16; // rdi
  unsigned __int64 v17; // rdi
  unsigned __int64 v18; // rdi
  __int64 v19; // rax
  __int64 v20; // rbx
  unsigned int v21; // r12d
  void *v22; // rax
  void *v23; // rcx
  unsigned __int64 v24; // rdi
  __int64 v26; // rsi
  volatile signed __int32 *v27; // rax
  __int64 v28; // rax
  volatile signed __int32 *v29; // rax
  __int64 v30; // [rsp+0h] [rbp-90h] BYREF
  volatile signed __int32 *v31; // [rsp+8h] [rbp-88h]
  __int64 v32; // [rsp+10h] [rbp-80h] BYREF
  volatile signed __int32 *v33; // [rsp+18h] [rbp-78h]
  __int64 v34; // [rsp+20h] [rbp-70h]
  int v35; // [rsp+28h] [rbp-68h]
  unsigned __int64 v36; // [rsp+30h] [rbp-60h]
  int v37; // [rsp+38h] [rbp-58h]
  __int64 v38; // [rsp+40h] [rbp-50h]
  volatile signed __int32 *v39; // [rsp+48h] [rbp-48h]
  char v40; // [rsp+50h] [rbp-40h]
  unsigned __int64 v41; // [rsp+58h] [rbp-38h]
  __int64 v42; // [rsp+60h] [rbp-30h]
  __int64 v43; // [rsp+68h] [rbp-28h]

  v3 = *a2;
  v4 = (volatile signed __int32 *)*((_QWORD *)a2 + 1);
  *a2 = 0;
  *((_QWORD *)a2 + 1) = 0;
  LODWORD(v32) = v3;
  v33 = v4;
  sub_35BE780(&v30, (__int64)(a1 + 11), (unsigned int *)&v32);
  if ( v33 )
    j_j___libc_free_0_0((unsigned __int64)v33);
  v5 = v31;
  v6 = v30;
  if ( v31 )
  {
    if ( &_pthread_key_create )
      _InterlockedAdd(v31 + 2, 1u);
    else
      ++*((_DWORD *)v31 + 2);
  }
  v32 = v6;
  v7 = a1[24];
  v33 = v5;
  v8 = a1[20];
  v34 = 0;
  v35 = 0;
  v36 = 0;
  v37 = 0;
  v38 = 0;
  v39 = 0;
  v40 = 0;
  v41 = 0;
  v42 = 0;
  v43 = 0;
  if ( v7 == a1[23] )
  {
    v26 = a1[21];
    v9 = -1431655765 * ((v26 - v8) >> 5);
    if ( v26 == a1[22] )
    {
      sub_35BD510(a1 + 20, (_QWORD *)v26, &v32);
      v18 = v41;
LABEL_15:
      if ( v18 )
        j_j___libc_free_0(v18);
      goto LABEL_17;
    }
    if ( v26 )
    {
      *(_QWORD *)(v26 + 8) = 0;
      *(_QWORD *)v26 = v6;
      v27 = v33;
      v33 = 0;
      *(_QWORD *)(v26 + 8) = v27;
      v32 = 0;
      *(_QWORD *)(v26 + 16) = v34;
      *(_DWORD *)(v26 + 24) = v35;
      *(_QWORD *)(v26 + 32) = v36;
      v36 = 0;
      *(_DWORD *)(v26 + 40) = v37;
      v28 = v38;
      *(_QWORD *)(v26 + 56) = 0;
      *(_QWORD *)(v26 + 48) = v28;
      v29 = v39;
      v39 = 0;
      *(_QWORD *)(v26 + 56) = v29;
      v38 = 0;
      *(_BYTE *)(v26 + 64) = v40;
      *(_QWORD *)(v26 + 72) = v41;
      *(_QWORD *)(v26 + 80) = v42;
      *(_QWORD *)(v26 + 88) = v43;
      v26 = a1[21];
    }
    a1[21] = v26 + 96;
  }
  else
  {
    v9 = *(_DWORD *)(v7 - 4);
    v33 = 0;
    a1[24] = v7 - 4;
    v32 = 0;
    v10 = v8 + 96LL * v9;
    *(_QWORD *)v10 = v6;
    v11 = *(volatile signed __int32 **)(v10 + 8);
    *(_QWORD *)(v10 + 8) = v5;
    if ( v11 )
      sub_A191D0(v11);
    *(_QWORD *)(v10 + 16) = v34;
    *(_DWORD *)(v10 + 24) = v35;
    v12 = v36;
    v36 = 0;
    v13 = *(_QWORD *)(v10 + 32);
    *(_QWORD *)(v10 + 32) = v12;
    if ( v13 )
      j_j___libc_free_0_0(v13);
    *(_DWORD *)(v10 + 40) = v37;
    v14 = v38;
    v15 = v39;
    v38 = 0;
    v39 = 0;
    v16 = *(volatile signed __int32 **)(v10 + 56);
    *(_QWORD *)(v10 + 48) = v14;
    *(_QWORD *)(v10 + 56) = v15;
    if ( v16 )
      sub_A191D0(v16);
    v17 = *(_QWORD *)(v10 + 72);
    *(_BYTE *)(v10 + 64) = v40;
    *(_QWORD *)(v10 + 72) = v41;
    *(_QWORD *)(v10 + 80) = v42;
    *(_QWORD *)(v10 + 88) = v43;
    v41 = 0;
    v42 = 0;
    v43 = 0;
    if ( v17 )
    {
      j_j___libc_free_0(v17);
      v18 = v41;
      goto LABEL_15;
    }
  }
LABEL_17:
  if ( v39 )
    sub_A191D0(v39);
  if ( v36 )
    j_j___libc_free_0_0(v36);
  if ( v33 )
    sub_A191D0(v33);
  v19 = a1[19];
  if ( v19 )
  {
    v20 = *(_QWORD *)(*(_QWORD *)v19 + 160LL) + 96LL * v9;
    v21 = **(_DWORD **)v20 - 1;
    *(_DWORD *)(v20 + 20) = v21;
    v22 = (void *)sub_2207820(4LL * v21);
    v23 = v22;
    if ( v22 && v21 )
      v23 = memset(v22, 0, 4LL * v21);
    v24 = *(_QWORD *)(v20 + 32);
    *(_QWORD *)(v20 + 32) = v23;
    if ( v24 )
      j_j___libc_free_0_0(v24);
  }
  if ( v31 )
    sub_A191D0(v31);
  return v9;
}
