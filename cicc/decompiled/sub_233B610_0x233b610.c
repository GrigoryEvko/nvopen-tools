// Function: sub_233B610
// Address: 0x233b610
//
__int64 __fastcall sub_233B610(__int64 a1)
{
  __int64 v1; // r15
  __int64 v2; // r12
  __int64 v3; // rax
  _QWORD *v4; // rbx
  _QWORD *v5; // r13
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rdi
  _QWORD *v10; // r13
  _QWORD *v11; // rbx
  _QWORD **v12; // r12
  _QWORD *v13; // r14
  __int64 v14; // r15
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rax
  unsigned __int64 v18; // rdi
  unsigned __int64 *v19; // rbx
  unsigned __int64 v20; // r12
  unsigned __int64 v21; // rdi
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v25; // rax
  _QWORD *v26; // rbx
  _QWORD *v27; // r12
  __int64 v28; // rax
  _QWORD *v29; // rbx
  _QWORD *v30; // r12
  __int64 v31; // rax
  __int64 i; // [rsp+8h] [rbp-98h]
  _QWORD *v33; // [rsp+10h] [rbp-90h]
  _QWORD *v35; // [rsp+20h] [rbp-80h]
  unsigned __int64 v36; // [rsp+28h] [rbp-78h]
  __int64 v37; // [rsp+30h] [rbp-70h] BYREF
  __int64 v38; // [rsp+38h] [rbp-68h]
  __int64 v39; // [rsp+40h] [rbp-60h]
  __int64 v40; // [rsp+50h] [rbp-50h] BYREF
  __int64 v41; // [rsp+58h] [rbp-48h]
  __int64 v42; // [rsp+60h] [rbp-40h]

  v1 = a1 + 144;
  v2 = a1 + 720;
  do
  {
    v3 = *(unsigned int *)(v2 + 24);
    if ( (_DWORD)v3 )
    {
      v4 = *(_QWORD **)(v2 + 8);
      v5 = &v4[9 * v3];
      do
      {
        while ( *v4 == -4096 )
        {
          if ( v4[1] != -4096 )
            goto LABEL_5;
          v4 += 9;
          if ( v5 == v4 )
            goto LABEL_15;
        }
        if ( *v4 != -8192 || v4[1] != -8192 )
        {
LABEL_5:
          v6 = v4[7];
          if ( v6 != 0 && v6 != -4096 && v6 != -8192 )
            sub_BD60C0(v4 + 5);
          v7 = v4[4];
          if ( v7 != 0 && v7 != -4096 && v7 != -8192 )
            sub_BD60C0(v4 + 2);
        }
        v4 += 9;
      }
      while ( v5 != v4 );
    }
LABEL_15:
    v8 = *(unsigned int *)(v2 + 24);
    v9 = *(_QWORD *)(v2 + 8);
    v2 -= 32;
    sub_C7D6A0(v9, 72 * v8, 8);
  }
  while ( v1 != v2 );
  v10 = *(_QWORD **)(a1 + 152);
  v11 = *(_QWORD **)(a1 + 112);
  v35 = *(_QWORD **)(a1 + 144);
  v33 = *(_QWORD **)(a1 + 128);
  v36 = *(_QWORD *)(a1 + 168);
  v12 = (_QWORD **)(*(_QWORD *)(a1 + 136) + 8LL);
  for ( i = *(_QWORD *)(a1 + 136); v36 > (unsigned __int64)v12; ++v12 )
  {
    v13 = *v12;
    v14 = (__int64)(*v12 + 63);
    do
    {
      v15 = v13[2];
      if ( v15 != 0 && v15 != -4096 && v15 != -8192 )
        sub_BD60C0(v13);
      v13 += 3;
    }
    while ( (_QWORD *)v14 != v13 );
  }
  if ( v36 == i )
  {
    while ( v35 != v11 )
    {
      v25 = v11[2];
      if ( v25 != -4096 && v25 != 0 && v25 != -8192 )
        sub_BD60C0(v11);
      v11 += 3;
    }
  }
  else
  {
    for ( ; v33 != v11; v11 += 3 )
    {
      v16 = v11[2];
      if ( v16 != 0 && v16 != -4096 && v16 != -8192 )
        sub_BD60C0(v11);
    }
    for ( ; v35 != v10; v10 += 3 )
    {
      v17 = v10[2];
      if ( v17 != 0 && v17 != -4096 && v17 != -8192 )
        sub_BD60C0(v10);
    }
  }
  v18 = *(_QWORD *)(a1 + 96);
  if ( v18 )
  {
    v19 = *(unsigned __int64 **)(a1 + 136);
    v20 = *(_QWORD *)(a1 + 168) + 8LL;
    if ( v20 > (unsigned __int64)v19 )
    {
      do
      {
        v21 = *v19++;
        j_j___libc_free_0(v21);
      }
      while ( v20 > (unsigned __int64)v19 );
      v18 = *(_QWORD *)(a1 + 96);
    }
    j_j___libc_free_0(v18);
  }
  v22 = *(unsigned int *)(a1 + 88);
  if ( (_DWORD)v22 )
  {
    v26 = *(_QWORD **)(a1 + 72);
    v37 = 0;
    v38 = 0;
    v39 = -4096;
    v27 = &v26[3 * v22];
    v40 = 0;
    v41 = 0;
    v42 = -8192;
    do
    {
      v28 = v26[2];
      if ( v28 != 0 && v28 != -4096 && v28 != -8192 )
        sub_BD60C0(v26);
      v26 += 3;
    }
    while ( v27 != v26 );
    sub_D68D70(&v40);
    sub_D68D70(&v37);
    v22 = *(unsigned int *)(a1 + 88);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 72), 24 * v22, 8);
  v23 = *(unsigned int *)(a1 + 56);
  if ( (_DWORD)v23 )
  {
    v29 = *(_QWORD **)(a1 + 40);
    v37 = 0;
    v38 = 0;
    v39 = -4096;
    v30 = &v29[4 * v23];
    v40 = 0;
    v41 = 0;
    v42 = -8192;
    do
    {
      v31 = v29[2];
      if ( v31 != 0 && v31 != -4096 && v31 != -8192 )
        sub_BD60C0(v29);
      v29 += 4;
    }
    while ( v30 != v29 );
    sub_D68D70(&v40);
    sub_D68D70(&v37);
    LODWORD(v23) = *(_DWORD *)(a1 + 56);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 40), 32LL * (unsigned int)v23, 8);
  return sub_C7D6A0(*(_QWORD *)(a1 + 8), 16LL * *(unsigned int *)(a1 + 24), 8);
}
