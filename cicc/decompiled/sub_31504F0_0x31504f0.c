// Function: sub_31504F0
// Address: 0x31504f0
//
unsigned __int64 *__fastcall sub_31504F0(unsigned __int64 *a1, unsigned __int32 a2, _BYTE *a3, __int32 a4, char a5)
{
  _DWORD *v7; // rax
  int *v8; // rax
  int v9; // eax
  _QWORD *v10; // rax
  _QWORD *v11; // rax
  __int64 v12; // rbx
  _QWORD *v13; // r14
  unsigned __int64 *v14; // rsi
  _QWORD *v15; // rbx
  _QWORD *v16; // rax
  __int64 v18; // rax
  _QWORD *v19; // rax
  _QWORD *v20; // rax
  _QWORD *v21; // rax
  _QWORD *v22; // rax
  int v23; // r15d
  __int64 v24; // rax
  unsigned __int64 v25; // rbx
  _QWORD *v26; // r14
  unsigned __int64 *v27; // rsi
  _QWORD *v28; // rbx
  int *v29; // rax
  int v30; // eax
  _DWORD *v31; // rax
  __int64 v32; // rax
  _QWORD *v33; // rax
  __int64 v34; // rax
  __int64 v35; // rax
  char v37; // [rsp+Ch] [rbp-384h]
  int v38; // [rsp+1Ch] [rbp-374h] BYREF
  __int64 v39[2]; // [rsp+20h] [rbp-370h] BYREF
  _QWORD *v40; // [rsp+30h] [rbp-360h] BYREF
  _QWORD *v41; // [rsp+38h] [rbp-358h]
  __int64 v42; // [rsp+40h] [rbp-350h]
  __int64 v43; // [rsp+48h] [rbp-348h]
  __int64 v44; // [rsp+50h] [rbp-340h]
  _QWORD *v45; // [rsp+60h] [rbp-330h] BYREF
  _QWORD *v46; // [rsp+68h] [rbp-328h]
  __int64 v47; // [rsp+70h] [rbp-320h]
  __int64 v48; // [rsp+78h] [rbp-318h]
  __int64 v49; // [rsp+80h] [rbp-310h]

  v7 = (_DWORD *)sub_CEECD0(4, 4u);
  *v7 = a2;
  sub_C94E10((__int64)qword_4F86370, v7);
  v8 = (int *)sub_C94E20((__int64)qword_4F86310);
  if ( v8 )
    v9 = *v8;
  else
    v9 = qword_4F86310[2];
  if ( v9 )
  {
    v29 = (int *)sub_C94E20((__int64)qword_4F86310);
    if ( v29 )
      v30 = *v29;
    else
      v30 = qword_4F86310[2];
    if ( v30 != 1 || a2 != 3 )
      goto LABEL_5;
  }
  else if ( a2 != 3 )
  {
    goto LABEL_5;
  }
  if ( !BYTE4(qword_4F862D0[2]) )
  {
    v31 = (_DWORD *)sub_CEECD0(4, 4u);
    *v31 = 6;
    sub_C94E10((__int64)qword_4F862D0, v31);
  }
LABEL_5:
  *a1 = 0;
  a1[1] = 0;
  a1[2] = 0;
  a1[3] = 0;
  a1[4] = 0;
  v10 = (_QWORD *)sub_22077B0(0x10u);
  if ( v10 )
    *v10 = &unk_4A0CDF8;
  v45 = v10;
  sub_314D9D0(a1, (unsigned __int64 *)&v45);
  if ( v45 )
    (*(void (__fastcall **)(_QWORD *))(*v45 + 8LL))(v45);
  v11 = (_QWORD *)sub_22077B0(0x10u);
  if ( v11 )
    *v11 = &unk_4A0E678;
  v45 = v11;
  sub_314D9D0(a1, (unsigned __int64 *)&v45);
  if ( v45 )
    (*(void (__fastcall **)(_QWORD *))(*v45 + 8LL))(v45);
  if ( a2 )
  {
    sub_314DBB0((unsigned __int64 *)&v45, a2, a3, a4, a5);
    v12 = (__int64)v45;
    v13 = v46;
    if ( v45 != v46 )
    {
      do
      {
        v14 = (unsigned __int64 *)v12;
        v12 += 8;
        sub_314D9D0(a1, v14);
      }
      while ( v13 != (_QWORD *)v12 );
      v15 = v46;
      v13 = v45;
      if ( v46 != v45 )
      {
        do
        {
          if ( *v13 )
            (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*v13 + 8LL))(*v13);
          ++v13;
        }
        while ( v15 != v13 );
        v13 = v45;
      }
    }
    if ( v13 )
      j_j___libc_free_0((unsigned __int64)v13);
    goto LABEL_23;
  }
  v40 = 0;
  v41 = 0;
  v42 = 0;
  v43 = 0;
  v44 = 0;
  v18 = sub_22077B0(0x10u);
  if ( v18 )
  {
    *(_BYTE *)(v18 + 8) = 0;
    *(_QWORD *)v18 = &unk_4A0E9F8;
  }
  v45 = (_QWORD *)v18;
  sub_314D9D0((unsigned __int64 *)&v40, (unsigned __int64 *)&v45);
  if ( v45 )
    (*(void (__fastcall **)(_QWORD *))(*v45 + 8LL))(v45);
  if ( *a3 )
  {
    v19 = (_QWORD *)sub_22077B0(0x10u);
    if ( !v19 )
      goto LABEL_35;
    goto LABEL_34;
  }
  v37 = a3[5];
  v32 = sub_22077B0(0x10u);
  if ( v32 )
  {
    *(_BYTE *)(v32 + 8) = v37;
    *(_QWORD *)v32 = &unk_4A11DB8;
  }
  v45 = (_QWORD *)v32;
  LOBYTE(v46) = 0;
  sub_23571D0((unsigned __int64 *)&v40, (__int64 *)&v45);
  sub_233EFE0((__int64 *)&v45);
  v19 = (_QWORD *)sub_22077B0(0x10u);
  if ( v19 )
LABEL_34:
    *v19 = &unk_4A0E5F8;
LABEL_35:
  v45 = v19;
  sub_314D9D0((unsigned __int64 *)&v40, (unsigned __int64 *)&v45);
  if ( v45 )
    (*(void (__fastcall **)(_QWORD *))(*v45 + 8LL))(v45);
  if ( a5 )
  {
    v20 = (_QWORD *)sub_22077B0(0x10u);
    if ( v20 )
      *v20 = &unk_4A0E4F8;
    v45 = v20;
    sub_314D9D0((unsigned __int64 *)&v40, (unsigned __int64 *)&v45);
    sub_23501E0((__int64 *)&v45);
    if ( a3[2] )
    {
      v45 = 0;
      v46 = 0;
      v47 = 0;
      v48 = 0;
      v49 = 0;
LABEL_42:
      sub_291E720(v39, 0);
      sub_23A2000((unsigned __int64 *)&v45, (char *)v39);
LABEL_43:
      if ( !a3[2] )
      {
        v21 = (_QWORD *)sub_22077B0(0x10u);
        if ( v21 )
          *v21 = &unk_4A0FFF8;
        v39[0] = (__int64)v21;
        sub_314D790((unsigned __int64 *)&v45, (unsigned __int64 *)v39);
        sub_233EFE0(v39);
      }
      goto LABEL_51;
    }
LABEL_76:
    v33 = (_QWORD *)sub_22077B0(0x10u);
    if ( v33 )
      *v33 = &unk_4A0E5B8;
    v45 = v33;
    sub_314D9D0((unsigned __int64 *)&v40, (unsigned __int64 *)&v45);
    sub_23501E0((__int64 *)&v45);
    v34 = sub_22077B0(0x10u);
    if ( v34 )
    {
      *(_BYTE *)(v34 + 8) = 1;
      *(_QWORD *)v34 = &unk_4A0CDB8;
    }
    v45 = (_QWORD *)v34;
    sub_314D9D0((unsigned __int64 *)&v40, (unsigned __int64 *)&v45);
    if ( v45 )
      (*(void (__fastcall **)(_QWORD *))(*v45 + 8LL))(v45);
    sub_23A0BA0((__int64)&v45, 0);
    sub_23A2670((unsigned __int64 *)&v40, (__int64)&v45);
    sub_233AAF0((__int64)&v45);
    v45 = 0;
    v46 = 0;
    v47 = 0;
    v48 = 0;
    v49 = 0;
    if ( !a5 )
      goto LABEL_43;
    goto LABEL_42;
  }
  v22 = (_QWORD *)sub_22077B0(0x10u);
  if ( v22 )
    *v22 = &unk_4A30EE0;
  v45 = v22;
  sub_314D9D0((unsigned __int64 *)&v40, (unsigned __int64 *)&v45);
  sub_23501E0((__int64 *)&v45);
  if ( !a3[2] )
    goto LABEL_76;
  v45 = 0;
  v46 = 0;
  v47 = 0;
  v48 = 0;
  v49 = 0;
LABEL_51:
  if ( a3[6] )
  {
    if ( (_BYTE)qword_5034328 )
    {
      sub_27D05A0(&v38);
      v23 = v38;
      v24 = sub_22077B0(0x10u);
      if ( v24 )
      {
        *(_DWORD *)(v24 + 8) = v23;
        *(_QWORD *)v24 = &unk_4A0F878;
      }
      v39[0] = v24;
      sub_314D790((unsigned __int64 *)&v45, (unsigned __int64 *)v39);
      sub_233EFE0(v39);
    }
    else
    {
      v35 = sub_22077B0(0x10u);
      if ( v35 )
      {
        *(_QWORD *)v35 = &unk_4A11D38;
        *(_WORD *)(v35 + 8) = 1;
      }
      v39[0] = v35;
      sub_314D790((unsigned __int64 *)&v45, (unsigned __int64 *)v39);
      if ( v39[0] )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v39[0] + 8LL))(v39[0]);
    }
  }
  sub_234AAB0((__int64)v39, (__int64 *)&v45, 0);
  sub_23571D0((unsigned __int64 *)&v40, v39);
  sub_233EFE0(v39);
  sub_233F7F0((__int64)&v45);
  v25 = (unsigned __int64)v40;
  v26 = v41;
  if ( v40 != v41 )
  {
    do
    {
      v27 = (unsigned __int64 *)v25;
      v25 += 8LL;
      sub_314D9D0(a1, v27);
    }
    while ( v26 != (_QWORD *)v25 );
    v28 = v41;
    v26 = v40;
    if ( v41 != v40 )
    {
      do
      {
        if ( *v26 )
          (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*v26 + 8LL))(*v26);
        ++v26;
      }
      while ( v28 != v26 );
      v26 = v40;
    }
  }
  if ( v26 )
    j_j___libc_free_0((unsigned __int64)v26);
LABEL_23:
  v16 = (_QWORD *)sub_22077B0(0x10u);
  if ( v16 )
    *v16 = &unk_4A0E538;
  v45 = v16;
  sub_314D9D0(a1, (unsigned __int64 *)&v45);
  if ( v45 )
    (*(void (__fastcall **)(_QWORD *))(*v45 + 8LL))(v45);
  return a1;
}
