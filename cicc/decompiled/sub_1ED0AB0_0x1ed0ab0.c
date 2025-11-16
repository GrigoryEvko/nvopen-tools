// Function: sub_1ED0AB0
// Address: 0x1ed0ab0
//
__int64 __fastcall sub_1ED0AB0(_QWORD *a1, int *a2)
{
  int v3; // edx
  volatile signed __int32 *v4; // rax
  volatile signed __int32 *v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // rax
  __int64 v8; // rdi
  unsigned int v9; // r13d
  __int64 v10; // rbx
  volatile signed __int32 *v11; // rdi
  __int64 v12; // rax
  __int64 v13; // rdi
  __int64 v14; // rdx
  volatile signed __int32 *v15; // rax
  volatile signed __int32 *v16; // rdi
  __int64 v17; // rdi
  __int64 v18; // rsi
  __int64 v19; // rdi
  __int64 v20; // rsi
  __int64 v21; // rax
  __int64 v22; // r12
  unsigned int v23; // ebx
  void *v24; // rax
  void *v25; // rcx
  __int64 v26; // rdi
  __int64 v28; // rsi
  volatile signed __int32 *v29; // rax
  __int64 v30; // rax
  volatile signed __int32 *v31; // rax
  __int64 v32; // [rsp+0h] [rbp-90h] BYREF
  volatile signed __int32 *v33; // [rsp+8h] [rbp-88h]
  __int64 v34; // [rsp+10h] [rbp-80h] BYREF
  volatile signed __int32 *v35; // [rsp+18h] [rbp-78h]
  __int64 v36; // [rsp+20h] [rbp-70h]
  int v37; // [rsp+28h] [rbp-68h]
  __int64 v38; // [rsp+30h] [rbp-60h]
  int v39; // [rsp+38h] [rbp-58h]
  __int64 v40; // [rsp+40h] [rbp-50h]
  volatile signed __int32 *v41; // [rsp+48h] [rbp-48h]
  __int64 v42; // [rsp+50h] [rbp-40h]
  __int64 v43; // [rsp+58h] [rbp-38h]
  __int64 v44; // [rsp+60h] [rbp-30h]

  v3 = *a2;
  v4 = (volatile signed __int32 *)*((_QWORD *)a2 + 1);
  *a2 = 0;
  *((_QWORD *)a2 + 1) = 0;
  LODWORD(v34) = v3;
  v35 = v4;
  sub_1ED0750(&v32, (__int64)(a1 + 11), (unsigned int *)&v34);
  if ( v35 )
    j_j___libc_free_0_0(v35);
  v5 = v33;
  v6 = v32;
  if ( v33 )
  {
    if ( &_pthread_key_create )
      _InterlockedAdd(v33 + 2, 1u);
    else
      ++*((_DWORD *)v33 + 2);
  }
  v34 = v6;
  v7 = a1[24];
  v35 = v5;
  v8 = a1[20];
  v36 = 0;
  v37 = 0;
  v38 = 0;
  v39 = 0;
  v40 = 0;
  v41 = 0;
  v42 = 0;
  v43 = 0;
  v44 = 0;
  if ( v7 == a1[23] )
  {
    v28 = a1[21];
    v9 = -1171354717 * ((v28 - v8) >> 3);
    if ( v28 == a1[22] )
    {
      sub_1ECF470(a1 + 20, (char *)v28, &v34);
      v19 = v42;
      v20 = v44 - v42;
LABEL_15:
      if ( v19 )
        j_j___libc_free_0(v19, v20);
      goto LABEL_17;
    }
    if ( v28 )
    {
      *(_QWORD *)(v28 + 8) = 0;
      *(_QWORD *)v28 = v6;
      v29 = v35;
      v35 = 0;
      *(_QWORD *)(v28 + 8) = v29;
      v34 = 0;
      *(_QWORD *)(v28 + 16) = v36;
      *(_DWORD *)(v28 + 24) = v37;
      *(_QWORD *)(v28 + 32) = v38;
      v38 = 0;
      *(_DWORD *)(v28 + 40) = v39;
      v30 = v40;
      *(_QWORD *)(v28 + 56) = 0;
      *(_QWORD *)(v28 + 48) = v30;
      v31 = v41;
      v41 = 0;
      *(_QWORD *)(v28 + 56) = v31;
      v40 = 0;
      *(_QWORD *)(v28 + 64) = v42;
      *(_QWORD *)(v28 + 72) = v43;
      *(_QWORD *)(v28 + 80) = v44;
      v28 = a1[21];
    }
    a1[21] = v28 + 88;
  }
  else
  {
    v9 = *(_DWORD *)(v7 - 4);
    v35 = 0;
    a1[24] = v7 - 4;
    v34 = 0;
    v10 = v8 + 88LL * v9;
    v11 = *(volatile signed __int32 **)(v10 + 8);
    *(_QWORD *)v10 = v6;
    *(_QWORD *)(v10 + 8) = v5;
    if ( v11 )
      sub_A191D0(v11);
    *(_QWORD *)(v10 + 16) = v36;
    *(_DWORD *)(v10 + 24) = v37;
    v12 = v38;
    v38 = 0;
    v13 = *(_QWORD *)(v10 + 32);
    *(_QWORD *)(v10 + 32) = v12;
    if ( v13 )
      j_j___libc_free_0_0(v13);
    *(_DWORD *)(v10 + 40) = v39;
    v14 = v40;
    v15 = v41;
    v40 = 0;
    v41 = 0;
    v16 = *(volatile signed __int32 **)(v10 + 56);
    *(_QWORD *)(v10 + 48) = v14;
    *(_QWORD *)(v10 + 56) = v15;
    if ( v16 )
      sub_A191D0(v16);
    v17 = *(_QWORD *)(v10 + 64);
    v18 = *(_QWORD *)(v10 + 80);
    *(_QWORD *)(v10 + 64) = v42;
    *(_QWORD *)(v10 + 72) = v43;
    *(_QWORD *)(v10 + 80) = v44;
    v42 = 0;
    v43 = 0;
    v44 = 0;
    if ( v17 )
    {
      j_j___libc_free_0(v17, v18 - v17);
      v19 = v42;
      v20 = v44 - v42;
      goto LABEL_15;
    }
  }
LABEL_17:
  if ( v41 )
    sub_A191D0(v41);
  if ( v38 )
    j_j___libc_free_0_0(v38);
  if ( v35 )
    sub_A191D0(v35);
  v21 = a1[19];
  if ( v21 )
  {
    v22 = *(_QWORD *)(*(_QWORD *)v21 + 160LL) + 88LL * v9;
    v23 = **(_DWORD **)v22 - 1;
    *(_DWORD *)(v22 + 20) = v23;
    v24 = (void *)sub_2207820(4LL * v23);
    v25 = v24;
    if ( v24 && v23 )
      v25 = memset(v24, 0, 4LL * v23);
    v26 = *(_QWORD *)(v22 + 32);
    *(_QWORD *)(v22 + 32) = v25;
    if ( v26 )
      j_j___libc_free_0_0(v26);
  }
  if ( v33 )
    sub_A191D0(v33);
  return v9;
}
