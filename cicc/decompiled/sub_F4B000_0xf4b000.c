// Function: sub_F4B000
// Address: 0xf4b000
//
unsigned __int64 __fastcall sub_F4B000(__int64 a1, _QWORD *a2, __int64 a3, char a4, __int64 a5, __int64 a6)
{
  unsigned __int8 v9; // r13
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // r10
  __int16 v14; // ax
  unsigned __int64 v15; // r15
  _QWORD *v16; // rax
  __int64 v17; // rcx
  _QWORD *i; // rdx
  __int64 v19; // rdx
  __int64 j; // r14
  __int64 v21; // r13
  __int64 v22; // rax
  unsigned __int64 v23; // rbx
  unsigned __int64 v24; // r13
  _QWORD *v25; // rax
  unsigned __int64 result; // rax
  _QWORD *v27; // rdi
  __int64 v28; // r14
  __int64 v29; // rcx
  __int64 v30; // r10
  _QWORD *v31; // [rsp+8h] [rbp-88h]
  __int64 v32; // [rsp+8h] [rbp-88h]
  __int64 v33; // [rsp+8h] [rbp-88h]
  _QWORD *v34; // [rsp+8h] [rbp-88h]
  __int64 v35; // [rsp+8h] [rbp-88h]
  __int64 v36; // [rsp+10h] [rbp-80h]
  __int64 v37; // [rsp+18h] [rbp-78h]
  unsigned __int64 v38; // [rsp+18h] [rbp-78h]
  __int64 v39; // [rsp+18h] [rbp-78h]
  __int64 v40; // [rsp+18h] [rbp-78h]
  __int64 v41; // [rsp+18h] [rbp-78h]
  __int64 v42; // [rsp+18h] [rbp-78h]
  __int64 v43; // [rsp+18h] [rbp-78h]
  __int64 v44; // [rsp+18h] [rbp-78h]
  __int64 v45; // [rsp+18h] [rbp-78h]
  __int64 v46; // [rsp+28h] [rbp-68h] BYREF
  _QWORD *v47; // [rsp+30h] [rbp-60h] BYREF
  __int64 v48; // [rsp+38h] [rbp-58h]
  _QWORD v49[10]; // [rsp+40h] [rbp-50h] BYREF

  v9 = a4 ^ 1;
  v36 = *(_QWORD *)(a1 + 120);
  sub_B2EC90(a1, (__int64)a2);
  v13 = (__int64)a2;
  *(_QWORD *)(a1 + 120) = v36;
  v14 = *((_WORD *)a2 + 1);
  if ( (v14 & 8) != 0 )
  {
    v34 = a2;
    v42 = sub_B2E500((__int64)a2);
    sub_FC75A0(&v47, a3, v9, a5, a6, 0);
    v43 = sub_FCD390(&v47, v42);
    sub_FC7680(&v47);
    a2 = (_QWORD *)v43;
    sub_B2E8C0(a1, v43);
    v13 = (__int64)v34;
    v14 = *((_WORD *)v34 + 1);
  }
  if ( (v14 & 2) != 0 )
  {
    v33 = v13;
    v40 = sub_B2E510(v13);
    sub_FC75A0(&v47, a3, v9, a5, a6, 0);
    v41 = sub_FCD390(&v47, v40);
    sub_FC7680(&v47);
    a2 = (_QWORD *)v41;
    sub_B2E9C0(a1, v41);
    v13 = v33;
    v14 = *(_WORD *)(v33 + 2);
  }
  if ( (v14 & 4) != 0 )
  {
    v32 = v13;
    v39 = sub_B2E520(v13);
    sub_FC75A0(&v47, a3, v9, a5, a6, 0);
    v28 = sub_FCD390(&v47, v39);
    sub_FC7680(&v47);
    a2 = (_QWORD *)v28;
    sub_B2EAD0(a1, v28);
    v13 = v32;
  }
  v15 = *(_QWORD *)(a1 + 104);
  v16 = v49;
  v17 = 0x400000000LL;
  i = v49;
  v47 = v49;
  v48 = 0x400000000LL;
  if ( v15 )
  {
    if ( v15 > 4 )
    {
      a2 = v49;
      v45 = v13;
      sub_C8D5F0((__int64)&v47, v49, v15, 8u, v11, v12);
      v13 = v45;
      v16 = &v47[(unsigned int)v48];
      for ( i = &v47[v15]; i != v16; ++v16 )
      {
LABEL_10:
        if ( v16 )
          *v16 = 0;
      }
    }
    else
    {
      i = &v49[v15];
      if ( i != v49 )
        goto LABEL_10;
    }
    LODWORD(v48) = v15;
  }
  v46 = *(_QWORD *)(v13 + 120);
  if ( (*(_BYTE *)(v13 + 2) & 1) != 0 )
  {
    v44 = v13;
    sub_B2C6D0(v13, (__int64)a2, (__int64)i, v17);
    v30 = v44;
    v19 = *(_QWORD *)(v44 + 96);
    v37 = v19 + 40LL * *(_QWORD *)(v44 + 104);
    if ( (*(_BYTE *)(v30 + 2) & 1) != 0 )
    {
      v35 = v30;
      sub_B2C6D0(v30, (__int64)a2, v19, v29);
      v19 = *(_QWORD *)(v35 + 96);
    }
  }
  else
  {
    v19 = *(_QWORD *)(v13 + 96);
    v37 = v19 + 40LL * *(_QWORD *)(v13 + 104);
  }
  for ( j = v19; v37 != j; j += 40 )
  {
    v21 = sub_F46C80(a3, j)[2];
    if ( *(_BYTE *)v21 == 22 )
    {
      v22 = sub_A744E0(&v46, *(_DWORD *)(j + 32));
      v47[*(unsigned int *)(v21 + 32)] = v22;
    }
  }
  v31 = v47;
  v38 = (unsigned int)v48;
  v23 = sub_A74610(&v46);
  v24 = sub_A74680(&v46);
  v25 = (_QWORD *)sub_B2BE50(a1);
  result = sub_A78180(v25, v24, v23, v31, v38);
  v27 = v47;
  *(_QWORD *)(a1 + 120) = result;
  if ( v27 != v49 )
    return _libc_free(v27, v24);
  return result;
}
