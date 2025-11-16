// Function: sub_1BE4260
// Address: 0x1be4260
//
void __fastcall sub_1BE4260(__int64 a1)
{
  _BYTE *v1; // rsi
  _QWORD *v2; // rdi
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // r8
  unsigned __int64 v6; // r13
  __int64 v7; // rax
  __int64 v8; // rdi
  __int64 v9; // rdx
  __int64 v10; // rax
  char v11; // si
  int v12; // r9d
  _BYTE *v13; // r8
  unsigned __int64 v14; // r13
  __int64 v15; // rax
  unsigned __int64 v16; // rdi
  unsigned __int64 v17; // rdx
  _BYTE *v18; // rax
  char v19; // cl
  unsigned __int64 v20; // rax
  __int64 v21; // r14
  __int64 v22; // rax
  __int64 v23; // rcx
  unsigned __int64 v24; // rax
  char v25; // si
  unsigned __int64 v26; // rsi
  _BYTE *v27; // rdi
  _QWORD *v28; // r13
  _QWORD *v29; // rbx
  _BYTE *v30; // [rsp+0h] [rbp-270h] BYREF
  __int64 v31; // [rsp+8h] [rbp-268h]
  _BYTE v32[64]; // [rsp+10h] [rbp-260h] BYREF
  _QWORD v33[2]; // [rsp+50h] [rbp-220h] BYREF
  unsigned __int64 v34; // [rsp+60h] [rbp-210h]
  _BYTE v35[64]; // [rsp+78h] [rbp-1F8h] BYREF
  __int64 v36; // [rsp+B8h] [rbp-1B8h]
  __int64 v37; // [rsp+C0h] [rbp-1B0h]
  unsigned __int64 v38; // [rsp+C8h] [rbp-1A8h]
  _QWORD v39[2]; // [rsp+D0h] [rbp-1A0h] BYREF
  unsigned __int64 v40; // [rsp+E0h] [rbp-190h]
  char v41[64]; // [rsp+F8h] [rbp-178h] BYREF
  unsigned __int64 v42; // [rsp+138h] [rbp-138h]
  unsigned __int64 i; // [rsp+140h] [rbp-130h]
  unsigned __int64 v44; // [rsp+148h] [rbp-128h]
  _QWORD v45[2]; // [rsp+150h] [rbp-120h] BYREF
  unsigned __int64 v46; // [rsp+160h] [rbp-110h]
  __int64 v47; // [rsp+1B8h] [rbp-B8h]
  __int64 v48; // [rsp+1C0h] [rbp-B0h]
  __int64 v49; // [rsp+1C8h] [rbp-A8h]
  char v50[8]; // [rsp+1D0h] [rbp-A0h] BYREF
  __int64 v51; // [rsp+1D8h] [rbp-98h]
  unsigned __int64 v52; // [rsp+1E0h] [rbp-90h]
  _BYTE *v53; // [rsp+238h] [rbp-38h]
  _BYTE *v54; // [rsp+240h] [rbp-30h]
  __int64 v55; // [rsp+248h] [rbp-28h]

  v31 = 0x800000000LL;
  v30 = v32;
  sub_1BE3E20(v45, a1);
  v1 = v35;
  v2 = v33;
  sub_16CCCB0(v33, (__int64)v35, (__int64)v45);
  v4 = v48;
  v5 = v47;
  v36 = 0;
  v37 = 0;
  v38 = 0;
  v6 = v48 - v47;
  if ( v48 == v47 )
  {
    v8 = 0;
  }
  else
  {
    if ( v6 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_64;
    v7 = sub_22077B0(v48 - v47);
    v4 = v48;
    v5 = v47;
    v8 = v7;
  }
  v36 = v8;
  v37 = v8;
  v38 = v8 + v6;
  if ( v4 != v5 )
  {
    v9 = v8;
    v10 = v5;
    do
    {
      if ( v9 )
      {
        *(_QWORD *)v9 = *(_QWORD *)v10;
        v11 = *(_BYTE *)(v10 + 16);
        *(_BYTE *)(v9 + 16) = v11;
        if ( v11 )
          *(_QWORD *)(v9 + 8) = *(_QWORD *)(v10 + 8);
      }
      v10 += 24;
      v9 += 24;
    }
    while ( v4 != v10 );
    v8 += 8 * ((unsigned __int64)(v4 - 24 - v5) >> 3) + 24;
  }
  v37 = v8;
  v2 = v39;
  sub_16CCCB0(v39, (__int64)v41, (__int64)v50);
  v1 = v54;
  v13 = v53;
  v42 = 0;
  i = 0;
  v44 = 0;
  v14 = v54 - v53;
  if ( v54 == v53 )
  {
    v16 = 0;
    goto LABEL_14;
  }
  if ( v14 > 0x7FFFFFFFFFFFFFF8LL )
LABEL_64:
    sub_4261EA(v2, v1, v3);
  v15 = sub_22077B0(v54 - v53);
  v1 = v54;
  v13 = v53;
  v16 = v15;
LABEL_14:
  v42 = v16;
  i = v16;
  v44 = v16 + v14;
  if ( v13 == v1 )
  {
    v20 = v16;
  }
  else
  {
    v17 = v16;
    v18 = v13;
    do
    {
      if ( v17 )
      {
        *(_QWORD *)v17 = *(_QWORD *)v18;
        v19 = v18[16];
        *(_BYTE *)(v17 + 16) = v19;
        if ( v19 )
          *(_QWORD *)(v17 + 8) = *((_QWORD *)v18 + 1);
      }
      v18 += 24;
      v17 += 24LL;
    }
    while ( v18 != v1 );
    v20 = v16 + 8 * ((unsigned __int64)(v18 - 24 - v13) >> 3) + 24;
  }
  for ( i = v20; ; v20 = i )
  {
    v23 = v36;
    if ( v37 - v36 != v20 - v16 )
      goto LABEL_22;
    if ( v36 == v37 )
      break;
    v24 = v16;
    while ( *(_QWORD *)v23 == *(_QWORD *)v24 )
    {
      v25 = *(_BYTE *)(v23 + 16);
      LODWORD(v13) = *(unsigned __int8 *)(v24 + 16);
      if ( v25 && (_BYTE)v13 )
      {
        if ( *(_QWORD *)(v23 + 8) != *(_QWORD *)(v24 + 8) )
          break;
        v23 += 24;
        v24 += 24LL;
        if ( v37 == v23 )
          goto LABEL_33;
      }
      else
      {
        if ( v25 != (_BYTE)v13 )
          break;
        v23 += 24;
        v24 += 24LL;
        if ( v37 == v23 )
          goto LABEL_33;
      }
    }
LABEL_22:
    v21 = *(_QWORD *)(v37 - 24);
    v22 = (unsigned int)v31;
    if ( (unsigned int)v31 >= HIDWORD(v31) )
    {
      sub_16CD150((__int64)&v30, v32, 0, 8, (int)v13, v12);
      v22 = (unsigned int)v31;
    }
    *(_QWORD *)&v30[8 * v22] = v21;
    LODWORD(v31) = v31 + 1;
    sub_1BE4140((__int64)v33);
    v16 = v42;
  }
LABEL_33:
  v26 = v44 - v16;
  if ( v16 )
    j_j___libc_free_0(v16, v26);
  if ( v40 != v39[1] )
    _libc_free(v40);
  if ( v36 )
  {
    v26 = v38 - v36;
    j_j___libc_free_0(v36, v38 - v36);
  }
  if ( v34 != v33[1] )
    _libc_free(v34);
  if ( v53 )
  {
    v26 = v55 - (_QWORD)v53;
    j_j___libc_free_0(v53, v55 - (_QWORD)v53);
  }
  if ( v52 != v51 )
    _libc_free(v52);
  if ( v47 )
  {
    v26 = v49 - v47;
    j_j___libc_free_0(v47, v49 - v47);
  }
  if ( v46 != v45[1] )
    _libc_free(v46);
  v27 = v30;
  v28 = &v30[8 * (unsigned int)v31];
  if ( v28 != (_QWORD *)v30 )
  {
    v29 = v30;
    do
    {
      if ( *v29 )
        (*(void (__fastcall **)(_QWORD, unsigned __int64))(*(_QWORD *)*v29 + 8LL))(*v29, v26);
      ++v29;
    }
    while ( v28 != v29 );
    v27 = v30;
  }
  if ( v27 != v32 )
    _libc_free((unsigned __int64)v27);
}
