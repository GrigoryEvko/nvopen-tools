// Function: sub_1B6CA70
// Address: 0x1b6ca70
//
__int64 __fastcall sub_1B6CA70(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 v5; // rax
  __int64 v6; // rdx
  unsigned __int64 v7; // rsi
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  unsigned __int64 *v11; // rbx
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  unsigned __int64 *v15; // rax
  __int64 *v16; // rdi
  __int64 v17; // r13
  int v18; // eax
  __int64 *v19; // r13
  __int64 *v20; // rax
  unsigned __int64 *v21; // r13
  unsigned __int64 v22; // rdi
  __int64 v23; // rdx
  unsigned __int64 *v24; // r13
  unsigned __int64 v25; // rdi
  unsigned __int64 v26; // rdi
  __int64 *v27; // rdi
  __int64 v28; // r13
  __int64 v29; // rax
  __int64 *v30; // r13
  __int64 *v31; // rax
  unsigned __int64 *v32; // r13
  unsigned __int64 v33; // rdi
  __int64 v34; // rdx
  unsigned __int64 *v35; // r13
  unsigned __int64 v36; // rdi
  __int64 v37; // rsi
  unsigned int v38; // r13d
  _QWORD *v39; // rbx
  _QWORD *v40; // r12
  __int64 *v41; // rbx
  __int64 *v42; // r12
  __int64 *v43; // rdi
  __int64 v45; // rax
  __int64 v46; // [rsp+8h] [rbp-B8h]
  __int64 *v47; // [rsp+10h] [rbp-B0h]
  unsigned __int64 *v48; // [rsp+18h] [rbp-A8h]
  unsigned __int64 v49; // [rsp+18h] [rbp-A8h]
  __int64 v50; // [rsp+18h] [rbp-A8h]
  unsigned __int64 *v51; // [rsp+18h] [rbp-A8h]
  __int64 v52[2]; // [rsp+20h] [rbp-A0h] BYREF
  _QWORD v53[2]; // [rsp+30h] [rbp-90h] BYREF
  char v54; // [rsp+40h] [rbp-80h]
  char v55; // [rsp+41h] [rbp-7Fh]
  __int64 *v56; // [rsp+50h] [rbp-70h] BYREF
  __int64 *v57; // [rsp+58h] [rbp-68h]
  __int64 v58; // [rsp+60h] [rbp-60h]
  _QWORD *v59; // [rsp+68h] [rbp-58h]
  _QWORD *v60; // [rsp+70h] [rbp-50h]
  __int64 v61; // [rsp+78h] [rbp-48h]
  __int64 v62; // [rsp+80h] [rbp-40h]
  __int64 v63; // [rsp+88h] [rbp-38h]

  v5 = *a2;
  v58 = 0;
  v59 = 0;
  v60 = 0;
  v61 = 0;
  v62 = 0;
  v63 = 0;
  v6 = *(_QWORD *)(v5 + 16);
  v57 = 0;
  v7 = *(_QWORD *)(v5 + 8);
  v56 = 0;
  sub_16FE750(v52, v7, v6 - v7, (__int64)&v56, 1u, 0);
  v11 = sub_16FF550((unsigned __int64 *)v52, v7, v8, v9, v10);
  v46 = sub_16F82C0();
  while ( 1 )
  {
    v15 = (unsigned __int64 *)v46;
    if ( v11 )
      break;
LABEL_21:
    while ( 2 )
    {
      if ( !v15 || !*v15 )
        goto LABEL_59;
      v16 = (__int64 *)*v11;
      v17 = *(_QWORD *)(*v11 + 112);
      if ( !v17 )
        goto LABEL_24;
LABEL_8:
      v18 = *(_DWORD *)(v17 + 32);
      if ( v18 )
        goto LABEL_35;
      if ( (unsigned __int8)sub_16FD950((__int64 **)v16, v7, v12, v13, v14) )
      {
LABEL_27:
        v28 = *(_QWORD *)*v11;
        v29 = sub_22077B0(168);
        if ( v29 )
        {
          v7 = v28;
          v50 = v29;
          sub_16FF2B0(v29, v28);
          v29 = v50;
        }
        v30 = (__int64 *)*v11;
        *v11 = v29;
        v47 = v30;
        if ( v30 )
        {
          sub_1B67E20(v30[17]);
          v31 = v30;
          v32 = (unsigned __int64 *)v30[3];
          v51 = &v32[*((unsigned int *)v31 + 8)];
          while ( v51 != v32 )
          {
            v33 = *v32++;
            _libc_free(v33);
          }
          v34 = 2LL * *((unsigned int *)v47 + 20);
          v35 = (unsigned __int64 *)v47[9];
          v49 = (unsigned __int64)&v35[v34];
          if ( v35 != &v35[v34] )
          {
            do
            {
              v36 = *v35;
              v35 += 2;
              _libc_free(v36);
            }
            while ( (unsigned __int64 *)v49 != v35 );
            goto LABEL_15;
          }
LABEL_16:
          if ( (__int64 *)v49 != v47 + 11 )
            _libc_free(v49);
          v26 = v47[3];
          if ( (__int64 *)v26 != v47 + 5 )
            _libc_free(v26);
          v7 = 168;
          j_j___libc_free_0(v47, 168);
          v15 = (unsigned __int64 *)v46;
          if ( !v11 )
            continue;
          goto LABEL_3;
        }
      }
      else
      {
LABEL_10:
        v19 = (__int64 *)*v11;
        *v11 = 0;
        v47 = v19;
        if ( v19 )
        {
          sub_1B67E20(v19[17]);
          v20 = v19;
          v21 = (unsigned __int64 *)v19[3];
          v48 = &v21[*((unsigned int *)v20 + 8)];
          while ( v48 != v21 )
          {
            v22 = *v21++;
            _libc_free(v22);
          }
          v23 = 2LL * *((unsigned int *)v47 + 20);
          v24 = (unsigned __int64 *)v47[9];
          v49 = (unsigned __int64)&v24[v23];
          if ( v24 == &v24[v23] )
            goto LABEL_16;
          do
          {
            v25 = *v24;
            v24 += 2;
            _libc_free(v25);
          }
          while ( (unsigned __int64 *)v49 != v24 );
LABEL_15:
          v49 = v47[9];
          goto LABEL_16;
        }
      }
      break;
    }
  }
LABEL_3:
  v16 = (__int64 *)*v11;
  if ( !*v11 )
    goto LABEL_21;
  if ( v15 && *v15 && v15 == v11 )
  {
LABEL_59:
    v38 = 1;
    goto LABEL_45;
  }
  v17 = v16[14];
  if ( v17 )
    goto LABEL_8;
LABEL_24:
  v17 = sub_16FC3B0((__int64)v16, v7, v12, v13, v14);
  v16[14] = v17;
  v18 = *(_DWORD *)(v17 + 32);
  if ( !v18 )
  {
LABEL_25:
    v27 = (__int64 *)*v11;
LABEL_26:
    if ( !(unsigned __int8)sub_16FD950((__int64 **)v27, v7, v12, v13, v14) )
      goto LABEL_10;
    goto LABEL_27;
  }
LABEL_35:
  if ( v18 == 4 )
  {
    *(_BYTE *)(v17 + 76) = 0;
    sub_16FD380(v17, v7);
    if ( !*(_QWORD *)(v17 + 80) )
      v17 = 0;
    while ( v17 )
    {
      v7 = (unsigned __int64)v52;
      if ( !(unsigned __int8)sub_1B6C890(a1, (unsigned __int64 *)v52, *(_QWORD *)(v17 + 80), a3, v14) )
        goto LABEL_44;
      sub_16FD380(v17, (unsigned __int64)v52);
      if ( !*(_QWORD *)(v17 + 80) )
      {
        v27 = (__int64 *)*v11;
        goto LABEL_26;
      }
    }
    goto LABEL_25;
  }
  v37 = v16[14];
  v55 = 1;
  v53[0] = "DescriptorList node must be a map";
  v54 = 3;
  if ( v37 )
  {
    sub_16F8270(v52, v37, (__int64)v53);
  }
  else
  {
    v45 = sub_16FC3B0((__int64)v16, 0, v12, v13, v14);
    v16[14] = v45;
    sub_16F8270(v52, v45, (__int64)v53);
  }
LABEL_44:
  v38 = 0;
LABEL_45:
  sub_16F8040(v52);
  v39 = v60;
  v40 = v59;
  if ( v60 != v59 )
  {
    do
    {
      if ( (_QWORD *)*v40 != v40 + 2 )
        j_j___libc_free_0(*v40, v40[2] + 1LL);
      v40 += 4;
    }
    while ( v39 != v40 );
    v40 = v59;
  }
  if ( v40 )
    j_j___libc_free_0(v40, v61 - (_QWORD)v40);
  v41 = v57;
  v42 = v56;
  if ( v57 != v56 )
  {
    do
    {
      v43 = v42;
      v42 += 3;
      sub_16CE300(v43);
    }
    while ( v41 != v42 );
    v42 = v56;
  }
  if ( v42 )
    j_j___libc_free_0(v42, v58 - (_QWORD)v42);
  return v38;
}
