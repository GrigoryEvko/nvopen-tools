// Function: sub_3530FF0
// Address: 0x3530ff0
//
__int64 __fastcall sub_3530FF0(__int64 a1, __int64 a2)
{
  unsigned int v2; // r13d
  __int64 (*v6)(); // rax
  __int64 v7; // rax
  __int64 v8; // rdi
  __int64 v9; // r15
  const char *v10; // rax
  size_t v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // rbx
  __int64 *v16; // r15
  __int64 v17; // r14
  __int64 v18; // rax
  unsigned __int64 v19; // rdx
  __int64 *v20; // r14
  __int64 (*v21)(); // rax
  __int64 (*v22)(); // rax
  __int64 *v23; // rax
  __int64 v24; // rsi
  __int64 v25; // rdx
  __int64 *v26; // rdx
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rax
  __int64 *v30; // rdx
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rax
  _DWORD *v34; // rax
  __int64 v35; // r14
  __int64 *v36; // rax
  __int64 v37; // rdx
  __int64 v38; // [rsp+10h] [rbp-A0h]
  char v39; // [rsp+18h] [rbp-98h]
  unsigned __int8 v40; // [rsp+20h] [rbp-90h]
  __int64 *v41; // [rsp+20h] [rbp-90h]
  __int64 v42; // [rsp+28h] [rbp-88h]
  __int64 v43; // [rsp+28h] [rbp-88h]
  char v44; // [rsp+3Fh] [rbp-71h] BYREF
  unsigned __int64 v45; // [rsp+40h] [rbp-70h] BYREF
  char v46; // [rsp+50h] [rbp-60h]
  __int64 *v47; // [rsp+60h] [rbp-50h] BYREF
  __int64 v48; // [rsp+68h] [rbp-48h]
  _BYTE v49[64]; // [rsp+70h] [rbp-40h] BYREF

  v2 = 0;
  if ( !*(_DWORD *)(*(_QWORD *)(a2 + 8) + 880LL) )
    return v2;
  sub_B2EE70((__int64)&v47, *(_QWORD *)a2, 0);
  v40 = v49[0];
  if ( !v49[0] && !(_BYTE)qword_503D588 )
    return 0;
  v6 = *(__int64 (**)())(**(_QWORD **)(a2 + 16) + 128LL);
  if ( v6 == sub_2DAC790 )
    BUG();
  v38 = v6();
  v2 = (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v38 + 1448LL))(v38, a2);
  if ( !(_BYTE)v2 )
    return 0;
  v7 = sub_B82360(*(_QWORD *)(a1 + 8), (__int64)&unk_501695C);
  if ( v7 )
  {
    v8 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v7 + 104LL))(v7, &unk_501695C);
    if ( v8 )
    {
      v9 = sub_2D514A0(v8);
      v10 = sub_2E791E0((__int64 *)a2);
      if ( (unsigned __int8)sub_2D51390(v9, v10, v11) )
        return 0;
    }
  }
  sub_2E7A760(a2, 0);
  *(_DWORD *)(a2 + 588) = 2;
  if ( v40 )
  {
    v26 = *(__int64 **)(a1 + 8);
    v27 = *v26;
    v28 = v26[1];
    if ( v27 == v28 )
LABEL_72:
      BUG();
    while ( *(_UNKNOWN **)v27 != &unk_501EC08 )
    {
      v27 += 16;
      if ( v28 == v27 )
        goto LABEL_72;
    }
    v29 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v27 + 8) + 104LL))(
            *(_QWORD *)(v27 + 8),
            &unk_501EC08);
    v30 = *(__int64 **)(a1 + 8);
    v16 = (__int64 *)(v29 + 200);
    v31 = *v30;
    v32 = v30[1];
    if ( v31 == v32 )
LABEL_73:
      BUG();
    while ( *(_UNKNOWN **)v31 != &unk_4F87C64 )
    {
      v31 += 16;
      if ( v32 == v31 )
        goto LABEL_73;
    }
    v33 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v31 + 8) + 104LL))(
            *(_QWORD *)(v31 + 8),
            &unk_4F87C64);
    v12 = a2 + 320;
    v15 = *(_QWORD *)(v33 + 176);
    v42 = a2 + 320;
    v34 = *(_DWORD **)(v15 + 8);
    if ( v34 )
    {
      if ( *v34 == 2 )
      {
        sub_2E7A420((__int64)&v45, v15, (__int64 *)a2);
        if ( !v46 || !sub_D84440(v15, v45) )
        {
          v35 = *(_QWORD *)(a2 + 328);
          if ( v35 == v42 )
          {
LABEL_63:
            if ( (_BYTE)qword_503D588 )
              sub_35303F0(a2);
            sub_34BCDA0(
              a2,
              (unsigned __int8 (__fastcall *)(__int64, __int64 *, unsigned __int64 *))sub_352F980,
              (__int64)&v47,
              v12,
              v13,
              v14);
            sub_34BC660(a2);
            return v40;
          }
          while ( 1 )
          {
            v36 = sub_2E39F50(v16, v35);
            v48 = v37;
            v47 = v36;
            if ( (_BYTE)v37 )
            {
              if ( sub_D84440(v15, (unsigned __int64)v47) )
                break;
            }
            v35 = *(_QWORD *)(v35 + 8);
            if ( v35 == v42 )
              goto LABEL_63;
          }
        }
      }
    }
  }
  else
  {
    v15 = 0;
    v16 = 0;
    v42 = a2 + 320;
  }
  v17 = *(_QWORD *)(a2 + 328);
  v47 = (__int64 *)v49;
  v48 = 0x200000000LL;
  if ( v17 == v42 )
  {
    if ( !(_BYTE)qword_503D588 )
      goto LABEL_34;
LABEL_60:
    sub_35303F0(a2);
    goto LABEL_34;
  }
  do
  {
    while ( sub_2E31AB0(v17) )
    {
LABEL_16:
      v17 = *(_QWORD *)(v17 + 8);
      if ( v17 == v42 )
        goto LABEL_22;
    }
    if ( !*(_BYTE *)(v17 + 216) )
    {
      if ( v40 )
      {
        if ( sub_352FA10(v17, v16, v15) )
        {
          v22 = *(__int64 (**)())(*(_QWORD *)v38 + 1456LL);
          if ( (v22 == sub_2FDC7E0 || ((unsigned __int8 (__fastcall *)(__int64, __int64))v22)(v38, v17))
            && !(_BYTE)qword_503D588 )
          {
            *(_QWORD *)(v17 + 252) = unk_501EB38;
          }
        }
      }
      goto LABEL_16;
    }
    v18 = (unsigned int)v48;
    v19 = (unsigned int)v48 + 1LL;
    if ( v19 > HIDWORD(v48) )
    {
      sub_C8D5F0((__int64)&v47, v49, v19, 8u, v13, v14);
      v18 = (unsigned int)v48;
    }
    v47[v18] = v17;
    LODWORD(v48) = v48 + 1;
    v17 = *(_QWORD *)(v17 + 8);
  }
  while ( v17 != v42 );
LABEL_22:
  v39 = qword_503D588;
  if ( (_BYTE)qword_503D588 )
    goto LABEL_60;
  v12 = (__int64)v47;
  v41 = &v47[(unsigned int)v48];
  if ( v47 != v41 )
  {
    v20 = v47;
    do
    {
      v43 = *v20;
      if ( !sub_352FA10(*v20, v16, v15)
        || (v13 = v43, v21 = *(__int64 (**)())(*(_QWORD *)v38 + 1456LL), v21 != sub_2FDC7E0)
        && !((unsigned __int8 (__fastcall *)(__int64, __int64))v21)(v38, v43) )
      {
        v39 = v2;
      }
      ++v20;
    }
    while ( v41 != v20 );
    if ( !v39 )
    {
      v23 = v47;
      v12 = (__int64)&v47[(unsigned int)v48];
      if ( v47 != (__int64 *)v12 )
      {
        v24 = unk_501EB38;
        do
        {
          v25 = *v23++;
          *(_QWORD *)(v25 + 252) = v24;
        }
        while ( v23 != (__int64 *)v12 );
      }
    }
  }
LABEL_34:
  sub_34BCDA0(
    a2,
    (unsigned __int8 (__fastcall *)(__int64, __int64 *, unsigned __int64 *))sub_352F980,
    (__int64)&v44,
    v12,
    v13,
    v14);
  sub_34BC660(a2);
  if ( v47 != (__int64 *)v49 )
    _libc_free((unsigned __int64)v47);
  return v2;
}
