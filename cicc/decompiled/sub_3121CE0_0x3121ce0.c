// Function: sub_3121CE0
// Address: 0x3121ce0
//
void __fastcall sub_3121CE0(__int64 a1, _BYTE *a2, size_t a3, unsigned int *a4)
{
  __int64 v7; // r14
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // r14
  __int64 v12; // rax
  __int64 v13; // rax
  _QWORD *v14; // rax
  __int64 v15; // r14
  __int64 v16; // rax
  __int64 v17; // rax
  const void *v18; // r8
  unsigned __int64 v19; // r13
  _QWORD *v20; // rax
  __int64 v21; // rdx
  _QWORD *v22; // rcx
  unsigned int **v23; // rbx
  unsigned int *v24; // rdi
  unsigned int *v25; // rdx
  _QWORD *v26; // rdi
  __int64 v27; // r13
  __int64 v28; // rax
  __int64 v29; // rax
  _QWORD *v30; // rdi
  size_t v31; // rdx
  void *src; // [rsp+8h] [rbp-D8h]
  size_t v33; // [rsp+18h] [rbp-C8h] BYREF
  __int64 v34[4]; // [rsp+20h] [rbp-C0h] BYREF
  _QWORD *v35; // [rsp+40h] [rbp-A0h] BYREF
  size_t n; // [rsp+48h] [rbp-98h]
  _QWORD v37[2]; // [rsp+50h] [rbp-90h] BYREF
  unsigned int *v38; // [rsp+60h] [rbp-80h]
  unsigned int *v39; // [rsp+68h] [rbp-78h]
  unsigned int *v40; // [rsp+70h] [rbp-70h]
  _QWORD *v41; // [rsp+78h] [rbp-68h] BYREF
  size_t v42; // [rsp+80h] [rbp-60h]
  _QWORD v43[2]; // [rsp+88h] [rbp-58h] BYREF
  unsigned int *v44; // [rsp+98h] [rbp-48h]
  unsigned int *v45; // [rsp+A0h] [rbp-40h]

  v7 = a4[8];
  v8 = sub_BCB2D0(*(_QWORD **)(**(_QWORD **)(a1 + 16) + 584LL));
  v9 = sub_ACD640(v8, v7, 0);
  v34[0] = (__int64)sub_B98A20(v9, v7);
  v10 = sub_B9B140(**(__int64 ***)(a1 + 24), a2, a3);
  v11 = a4[6];
  v34[1] = v10;
  v12 = sub_BCB2D0(*(_QWORD **)(**(_QWORD **)(a1 + 16) + 584LL));
  v13 = sub_ACD640(v12, v11, 0);
  v14 = sub_B98A20(v13, v11);
  v15 = a4[7];
  v34[2] = (__int64)v14;
  v16 = sub_BCB2D0(*(_QWORD **)(**(_QWORD **)(a1 + 16) + 584LL));
  v17 = sub_ACD640(v16, v15, 0);
  v34[3] = (__int64)sub_B98A20(v17, v15);
  v35 = v37;
  sub_3120C40((__int64 *)&v35, a2, (__int64)&a2[a3]);
  v18 = v35;
  v19 = n;
  v38 = 0;
  v39 = 0;
  v40 = a4;
  v41 = v43;
  if ( (_QWORD *)((char *)v35 + n) && !v35 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v33 = n;
  if ( n > 0xF )
  {
    src = v35;
    v29 = sub_22409D0((__int64)&v41, &v33, 0);
    v18 = src;
    v41 = (_QWORD *)v29;
    v30 = (_QWORD *)v29;
    v43[0] = v33;
  }
  else
  {
    if ( n == 1 )
    {
      LOBYTE(v43[0]) = *(_BYTE *)v35;
      v20 = v43;
      goto LABEL_6;
    }
    if ( !n )
    {
      v20 = v43;
      goto LABEL_6;
    }
    v30 = v43;
  }
  memcpy(v30, v18, v19);
  v19 = v33;
  v20 = v41;
LABEL_6:
  v42 = v19;
  *((_BYTE *)v20 + v19) = 0;
  v21 = a4[7];
  v22 = *(_QWORD **)(a1 + 8);
  v44 = v38;
  v45 = v39;
  v23 = (unsigned int **)(*v22 + 56 * v21);
  v24 = v23[1];
  *v23 = v40;
  if ( v41 == v43 )
  {
    v31 = v42;
    if ( v42 )
    {
      if ( v42 == 1 )
        *(_BYTE *)v24 = v43[0];
      else
        memcpy(v24, v43, v42);
      v31 = v42;
      v24 = v23[1];
    }
    v23[2] = (unsigned int *)v31;
    *((_BYTE *)v24 + v31) = 0;
    v24 = (unsigned int *)v41;
  }
  else
  {
    if ( v24 == (unsigned int *)(v23 + 3) )
    {
      v23[1] = (unsigned int *)v41;
      v23[2] = (unsigned int *)v42;
      v23[3] = (unsigned int *)v43[0];
    }
    else
    {
      v23[1] = (unsigned int *)v41;
      v25 = v23[3];
      v23[2] = (unsigned int *)v42;
      v23[3] = (unsigned int *)v43[0];
      if ( v24 )
      {
        v41 = v24;
        v43[0] = v25;
        goto LABEL_10;
      }
    }
    v41 = v43;
    v24 = (unsigned int *)v43;
  }
LABEL_10:
  v42 = 0;
  *(_BYTE *)v24 = 0;
  v26 = v41;
  v23[5] = v44;
  v23[6] = v45;
  if ( v26 != v43 )
    j_j___libc_free_0((unsigned __int64)v26);
  v27 = *(_QWORD *)(a1 + 32);
  v28 = sub_B9C770(*(__int64 **)a1, v34, (__int64 *)4, 0, 1);
  sub_B979A0(v27, v28);
  if ( v35 != v37 )
    j_j___libc_free_0((unsigned __int64)v35);
}
