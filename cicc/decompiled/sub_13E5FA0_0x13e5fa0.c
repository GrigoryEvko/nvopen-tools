// Function: sub_13E5FA0
// Address: 0x13e5fa0
//
__int64 __fastcall sub_13E5FA0(_QWORD *a1, __int64 a2)
{
  unsigned __int64 v3; // rsi
  __int64 v4; // rsi
  __int64 v5; // rbx
  unsigned __int64 *v6; // r14
  unsigned __int64 *v7; // rax
  char v8; // r8
  __int64 v9; // rbx
  _QWORD *v10; // r14
  __int64 v11; // rdi
  __int64 v12; // rdi
  __int64 v13; // rdi
  __int64 v14; // rdx
  __int64 v15; // rsi
  __int64 v16; // rbx
  unsigned __int64 *v17; // r14
  unsigned __int64 *v18; // rax
  char v19; // r8
  __int64 v20; // rbx
  _QWORD *v21; // r14
  __int64 v22; // rdi
  __int64 v23; // rdi
  __int64 v24; // rdi
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rbx
  __int64 v28; // r12
  __int64 v29; // rbx
  __int64 v30; // rdi
  _QWORD *v32; // r12
  __int64 v33; // rdi
  __int64 v34; // rdi
  __int64 v35; // rdi
  __int64 v36; // [rsp+0h] [rbp-E0h] BYREF
  __int64 v37; // [rsp+8h] [rbp-D8h]
  __int64 v38; // [rsp+10h] [rbp-D0h]
  int v39; // [rsp+20h] [rbp-C0h] BYREF
  __int64 v40; // [rsp+28h] [rbp-B8h]
  int *v41; // [rsp+30h] [rbp-B0h]
  int *v42; // [rsp+38h] [rbp-A8h]
  __int64 v43; // [rsp+40h] [rbp-A0h]
  __int64 v44; // [rsp+48h] [rbp-98h]
  char v45; // [rsp+50h] [rbp-90h]
  _QWORD v46[16]; // [rsp+60h] [rbp-80h] BYREF

  v44 = a2;
  v3 = *(_QWORD *)(a2 + 80);
  v41 = &v39;
  v42 = &v39;
  if ( v3 )
    v3 -= 24LL;
  v36 = 0;
  v37 = 0;
  v38 = 0;
  v39 = 0;
  v40 = 0;
  v43 = 0;
  v45 = 0;
  sub_13E5D90(&v36, v3);
  v4 = *(_QWORD *)(v37 - 16);
  a1[26] = v4;
  sub_13E5570(a1, v4);
  v5 = v37;
LABEL_4:
  v6 = *(unsigned __int64 **)(*(_QWORD *)(v5 - 16) + 40LL);
  v7 = *(unsigned __int64 **)(v5 - 8);
  do
  {
    if ( v6 == v7 )
    {
      v9 = v37;
      if ( v45 )
      {
        v10 = *(_QWORD **)(v37 - 16);
        if ( v10 )
        {
          v11 = v10[7];
          if ( v11 )
            j_j___libc_free_0(v11, v10[9] - v11);
          v12 = v10[4];
          if ( v12 )
            j_j___libc_free_0(v12, v10[6] - v12);
          v13 = v10[1];
          if ( v13 )
            j_j___libc_free_0(v13, v10[3] - v13);
          j_j___libc_free_0(v10, 80);
          v9 = v37;
        }
      }
      v14 = v36;
      v5 = v9 - 16;
      v37 = v5;
      if ( v5 == v36 )
      {
        v15 = v36;
        goto LABEL_19;
      }
      goto LABEL_4;
    }
    v8 = sub_13E5D90(&v36, *v7);
    v7 = (unsigned __int64 *)(*(_QWORD *)(v5 - 8) + 8LL);
    *(_QWORD *)(v5 - 8) = v7;
  }
  while ( !v8 );
  v14 = v37;
  v15 = v36;
LABEL_19:
  memset(v46, 0, 0x58u);
  v46[6] = &v46[4];
  v46[7] = &v46[4];
LABEL_20:
  if ( v14 != v15 )
  {
    sub_13E5570(a1, *(_QWORD *)(v14 - 16));
    v16 = v37;
    do
    {
      v17 = *(unsigned __int64 **)(*(_QWORD *)(v16 - 16) + 40LL);
      v18 = *(unsigned __int64 **)(v16 - 8);
      while ( v17 != v18 )
      {
        v19 = sub_13E5D90(&v36, *v18);
        v18 = (unsigned __int64 *)(*(_QWORD *)(v16 - 8) + 8LL);
        *(_QWORD *)(v16 - 8) = v18;
        if ( v19 )
        {
          v14 = v37;
          v15 = v36;
          goto LABEL_20;
        }
      }
      v20 = v37;
      if ( v45 )
      {
        v21 = *(_QWORD **)(v37 - 16);
        if ( v21 )
        {
          v22 = v21[7];
          if ( v22 )
            j_j___libc_free_0(v22, v21[9] - v22);
          v23 = v21[4];
          if ( v23 )
            j_j___libc_free_0(v23, v21[6] - v23);
          v24 = v21[1];
          if ( v24 )
            j_j___libc_free_0(v24, v21[3] - v24);
          j_j___libc_free_0(v21, 80);
          v20 = v37;
        }
      }
      v16 = v20 - 16;
      v37 = v16;
    }
    while ( v16 != v36 );
  }
  v25 = a1[27];
  v26 = (a1[28] - v25) >> 3;
  if ( (_DWORD)v26 )
  {
    v27 = 0;
    v28 = 8LL * (unsigned int)(v26 - 1);
    while ( 1 )
    {
      sub_13E5300((__int64)a1, *(char ***)(v25 + v27));
      if ( v27 == v28 )
        break;
      v25 = a1[27];
      v27 += 8;
    }
    if ( v45 )
    {
      while ( v37 != v36 )
      {
        v32 = *(_QWORD **)(v37 - 16);
        if ( v32 )
        {
          v33 = v32[7];
          if ( v33 )
            j_j___libc_free_0(v33, v32[9] - v33);
          v34 = v32[4];
          if ( v34 )
            j_j___libc_free_0(v34, v32[6] - v34);
          v35 = v32[1];
          if ( v35 )
            j_j___libc_free_0(v35, v32[3] - v35);
          j_j___libc_free_0(v32, 80);
        }
        v37 -= 16;
      }
    }
  }
  v29 = v40;
  while ( v29 )
  {
    sub_13E4AE0(*(_QWORD *)(v29 + 24));
    v30 = v29;
    v29 = *(_QWORD *)(v29 + 16);
    j_j___libc_free_0(v30, 40);
  }
  if ( v36 )
    j_j___libc_free_0(v36, v38 - v36);
  return 0;
}
