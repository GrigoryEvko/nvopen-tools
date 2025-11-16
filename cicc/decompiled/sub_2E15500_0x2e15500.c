// Function: sub_2E15500
// Address: 0x2e15500
//
void __fastcall sub_2E15500(unsigned int *a1, __int64 a2)
{
  __int64 v3; // r9
  __int64 (*v4)(void); // rdx
  bool v5; // zf
  unsigned __int64 v6; // rdx
  unsigned __int64 v7; // r8
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // rbx
  __int64 v17; // r13
  _QWORD *v18; // rax
  _QWORD *v19; // rcx
  __int64 v20; // rax
  _QWORD *v21; // rdx
  unsigned __int64 v22; // r13
  unsigned __int64 v23; // rdi
  unsigned __int64 v24; // rdi
  __int64 v25; // rax
  _QWORD *v26; // rbx
  _QWORD *v27; // r14
  unsigned __int64 v28; // rdi
  unsigned __int64 v29; // rdi
  unsigned __int64 v30; // rdi

  *(_QWORD *)a1 = a2;
  *((_QWORD *)a1 + 1) = *(_QWORD *)(a2 + 32);
  *((_QWORD *)a1 + 2) = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a2 + 16) + 200LL))(*(_QWORD *)(a2 + 16));
  v4 = *(__int64 (**)(void))(**(_QWORD **)(*(_QWORD *)a1 + 16LL) + 128LL);
  if ( v4 == sub_2DAC790 )
  {
    v5 = *((_QWORD *)a1 + 6) == 0;
    *((_QWORD *)a1 + 3) = 0;
    if ( !v5 )
      goto LABEL_3;
  }
  else
  {
    v20 = v4();
    v5 = *((_QWORD *)a1 + 6) == 0;
    *((_QWORD *)a1 + 3) = v20;
    if ( !v5 )
      goto LABEL_3;
  }
  v21 = (_QWORD *)sub_22077B0(0x2C8u);
  if ( v21 )
  {
    memset(v21, 0, 0x2C8u);
    v21[5] = v21 + 7;
    v21[6] = 0x600000000LL;
    v21[18] = v21 + 20;
    v21[23] = v21 + 25;
    v21[24] = 0x1000000000LL;
  }
  v22 = *((_QWORD *)a1 + 6);
  *((_QWORD *)a1 + 6) = v21;
  if ( v22 )
  {
    v23 = *(_QWORD *)(v22 + 184);
    if ( v23 != v22 + 200 )
      _libc_free(v23);
    v24 = *(_QWORD *)(v22 + 144);
    if ( v24 != v22 + 160 )
      _libc_free(v24);
    v25 = *(unsigned int *)(v22 + 136);
    if ( (_DWORD)v25 )
    {
      v26 = *(_QWORD **)(v22 + 120);
      v27 = &v26[19 * v25];
      do
      {
        if ( *v26 != -4096 && *v26 != -8192 )
        {
          v28 = v26[10];
          if ( (_QWORD *)v28 != v26 + 12 )
            _libc_free(v28);
          v29 = v26[1];
          if ( (_QWORD *)v29 != v26 + 3 )
            _libc_free(v29);
        }
        v26 += 19;
      }
      while ( v27 != v26 );
      v25 = *(unsigned int *)(v22 + 136);
    }
    sub_C7D6A0(*(_QWORD *)(v22 + 120), 152 * v25, 8);
    v30 = *(_QWORD *)(v22 + 40);
    if ( v30 != v22 + 56 )
      _libc_free(v30);
    a2 = 712;
    j_j___libc_free_0(v22);
  }
LABEL_3:
  v6 = a1[40];
  v7 = *(unsigned int *)(*((_QWORD *)a1 + 1) + 64LL);
  if ( v7 != v6 )
  {
    if ( v7 >= v6 )
    {
      v16 = *((_QWORD *)a1 + 21);
      v17 = v7 - v6;
      if ( v7 > a1[41] )
      {
        a2 = (__int64)(a1 + 42);
        sub_C8D5F0((__int64)(a1 + 38), a1 + 42, v7, 8u, v7, v3);
        v6 = a1[40];
      }
      v18 = (_QWORD *)(*((_QWORD *)a1 + 19) + 8 * v6);
      v19 = &v18[v17];
      if ( v18 != v19 )
      {
        do
          *v18++ = v16;
        while ( v19 != v18 );
        LODWORD(v6) = a1[40];
      }
      a1[40] = v17 + v6;
    }
    else
    {
      a1[40] = v7;
    }
  }
  sub_2E15330((__int64)a1);
  sub_2E10FB0(a1, a2, v8, v9, v10, v11);
  sub_2E11940(a1, a2, v12, v13, v14, v15);
}
