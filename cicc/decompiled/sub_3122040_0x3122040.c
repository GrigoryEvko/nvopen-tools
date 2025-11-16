// Function: sub_3122040
// Address: 0x3122040
//
void __fastcall sub_3122040(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v6; // r13
  __int64 v7; // rax
  __int64 v8; // rax
  _QWORD *v9; // rax
  __int64 v10; // r13
  __int64 v11; // rax
  __int64 v12; // rax
  _QWORD *v13; // rax
  __int64 v14; // r13
  __int64 v15; // rax
  __int64 v16; // rax
  _QWORD *v17; // rax
  size_t v18; // rdx
  _BYTE *v19; // rsi
  __int64 v20; // rax
  __int64 v21; // r13
  __int64 v22; // rax
  __int64 v23; // rax
  _QWORD *v24; // rax
  __int64 v25; // r13
  __int64 v26; // rax
  __int64 v27; // rax
  _QWORD *v28; // rax
  __int64 v29; // r13
  __int64 v30; // rax
  __int64 v31; // rax
  _QWORD *v32; // rax
  _BYTE *v33; // r8
  size_t v34; // r13
  _QWORD *v35; // rax
  __int64 v36; // rdx
  _QWORD *v37; // rcx
  _QWORD *v38; // rbx
  _BYTE *v39; // rdi
  __int64 v40; // rdx
  _QWORD *v41; // rdi
  __int64 v42; // r13
  __int64 v43; // rax
  __int64 v44; // rax
  _QWORD *v45; // rdi
  size_t v46; // rdx
  _BYTE *src; // [rsp+8h] [rbp-C8h]
  size_t v48; // [rsp+18h] [rbp-B8h] BYREF
  __int64 v49[8]; // [rsp+20h] [rbp-B0h] BYREF
  __int64 v50; // [rsp+60h] [rbp-70h]
  _QWORD *v51; // [rsp+68h] [rbp-68h] BYREF
  size_t n; // [rsp+70h] [rbp-60h]
  _QWORD v53[2]; // [rsp+78h] [rbp-58h] BYREF
  __int64 v54; // [rsp+88h] [rbp-48h]
  __int64 v55; // [rsp+90h] [rbp-40h]

  v6 = *(unsigned int *)(a3 + 32);
  v7 = sub_BCB2D0(*(_QWORD **)(**(_QWORD **)(a1 + 24) + 584LL));
  v8 = sub_ACD640(v7, v6, 0);
  v9 = sub_B98A20(v8, v6);
  v10 = *(unsigned int *)(a2 + 32);
  v49[0] = (__int64)v9;
  v11 = sub_BCB2D0(*(_QWORD **)(**(_QWORD **)(a1 + 24) + 584LL));
  v12 = sub_ACD640(v11, v10, 0);
  v13 = sub_B98A20(v12, v10);
  v14 = *(unsigned int *)(a2 + 36);
  v49[1] = (__int64)v13;
  v15 = sub_BCB2D0(*(_QWORD **)(**(_QWORD **)(a1 + 24) + 584LL));
  v16 = sub_ACD640(v15, v14, 0);
  v17 = sub_B98A20(v16, v14);
  v18 = *(_QWORD *)(a2 + 8);
  v19 = *(_BYTE **)a2;
  v49[2] = (__int64)v17;
  v20 = sub_B9B140(**(__int64 ***)(a1 + 32), v19, v18);
  v21 = *(unsigned int *)(a2 + 40);
  v49[3] = v20;
  v22 = sub_BCB2D0(*(_QWORD **)(**(_QWORD **)(a1 + 24) + 584LL));
  v23 = sub_ACD640(v22, v21, 0);
  v24 = sub_B98A20(v23, v21);
  v25 = *(unsigned int *)(a2 + 44);
  v49[4] = (__int64)v24;
  v26 = sub_BCB2D0(*(_QWORD **)(**(_QWORD **)(a1 + 24) + 584LL));
  v27 = sub_ACD640(v26, v25, 0);
  v28 = sub_B98A20(v27, v25);
  v29 = *(unsigned int *)(a3 + 28);
  v49[5] = (__int64)v28;
  v30 = sub_BCB2D0(*(_QWORD **)(**(_QWORD **)(a1 + 24) + 584LL));
  v31 = sub_ACD640(v30, v29, 0);
  v32 = sub_B98A20(v31, v29);
  v33 = *(_BYTE **)a2;
  v34 = *(_QWORD *)(a2 + 8);
  v50 = a3;
  v49[6] = (__int64)v32;
  v51 = v53;
  if ( &v33[v34] && !v33 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v48 = v34;
  if ( v34 > 0xF )
  {
    src = v33;
    v44 = sub_22409D0((__int64)&v51, &v48, 0);
    v33 = src;
    v51 = (_QWORD *)v44;
    v45 = (_QWORD *)v44;
    v53[0] = v48;
  }
  else
  {
    if ( v34 == 1 )
    {
      LOBYTE(v53[0]) = *v33;
      v35 = v53;
      goto LABEL_6;
    }
    if ( !v34 )
    {
      v35 = v53;
      goto LABEL_6;
    }
    v45 = v53;
  }
  memcpy(v45, v33, v34);
  v34 = v48;
  v35 = v51;
LABEL_6:
  n = v34;
  *((_BYTE *)v35 + v34) = 0;
  v36 = *(unsigned int *)(a3 + 28);
  v37 = *(_QWORD **)(a1 + 16);
  v54 = *(_QWORD *)(a2 + 32);
  v55 = *(_QWORD *)(a2 + 40);
  v38 = (_QWORD *)(*v37 + 56 * v36);
  v39 = (_BYTE *)v38[1];
  *v38 = v50;
  if ( v51 == v53 )
  {
    v46 = n;
    if ( n )
    {
      if ( n == 1 )
        *v39 = v53[0];
      else
        memcpy(v39, v53, n);
      v46 = n;
      v39 = (_BYTE *)v38[1];
    }
    v38[2] = v46;
    v39[v46] = 0;
    v39 = v51;
  }
  else
  {
    if ( v39 == (_BYTE *)(v38 + 3) )
    {
      v38[1] = v51;
      v38[2] = n;
      v38[3] = v53[0];
    }
    else
    {
      v38[1] = v51;
      v40 = v38[3];
      v38[2] = n;
      v38[3] = v53[0];
      if ( v39 )
      {
        v51 = v39;
        v53[0] = v40;
        goto LABEL_10;
      }
    }
    v51 = v53;
    v39 = v53;
  }
LABEL_10:
  n = 0;
  *v39 = 0;
  v41 = v51;
  v38[5] = v54;
  v38[6] = v55;
  if ( v41 != v53 )
    j_j___libc_free_0((unsigned __int64)v41);
  v42 = *(_QWORD *)(a1 + 8);
  v43 = sub_B9C770(*(__int64 **)a1, v49, (__int64 *)7, 0, 1);
  sub_B979A0(v42, v43);
}
