// Function: sub_AB3BD0
// Address: 0xab3bd0
//
__int64 __fastcall sub_AB3BD0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v5; // rbx
  __int64 v6; // rsi
  unsigned int v7; // r15d
  bool v8; // cf
  __int64 v9; // rax
  __int64 v10; // rax
  unsigned int *v12; // rdi
  unsigned int v13; // [rsp+10h] [rbp-C0h]
  unsigned int v14; // [rsp+10h] [rbp-C0h]
  unsigned int v16; // [rsp+18h] [rbp-B8h]
  __int64 v17; // [rsp+20h] [rbp-B0h] BYREF
  unsigned int v18; // [rsp+28h] [rbp-A8h]
  __int64 v19; // [rsp+30h] [rbp-A0h] BYREF
  unsigned int v20; // [rsp+38h] [rbp-98h]
  __int64 v21; // [rsp+40h] [rbp-90h] BYREF
  unsigned int v22; // [rsp+48h] [rbp-88h]
  __int64 v23; // [rsp+50h] [rbp-80h] BYREF
  unsigned int v24; // [rsp+58h] [rbp-78h]
  __int64 v25; // [rsp+60h] [rbp-70h] BYREF
  unsigned int v26; // [rsp+68h] [rbp-68h]
  __int64 v27; // [rsp+70h] [rbp-60h]
  unsigned int v28; // [rsp+78h] [rbp-58h]
  unsigned int *v29; // [rsp+80h] [rbp-50h] BYREF
  unsigned int v30; // [rsp+88h] [rbp-48h]
  __int64 v31; // [rsp+90h] [rbp-40h]
  unsigned int v32; // [rsp+98h] [rbp-38h]

  v5 = *(unsigned int *)(a2 + 8);
  sub_AB0A00((__int64)&v29, a3);
  v6 = a3;
  v7 = v5;
  v13 = v30;
  if ( v30 > 0x40 )
  {
    v6 = a3;
    if ( v13 - (unsigned int)sub_C444A0(&v29) > 0x40 )
    {
      v12 = v29;
      if ( !v29 )
        goto LABEL_3;
    }
    else
    {
      v12 = v29;
      if ( v5 >= *(_QWORD *)v29 )
      {
        v16 = *v29;
LABEL_23:
        j_j___libc_free_0_0(v12);
        goto LABEL_4;
      }
    }
    v16 = v5;
    goto LABEL_23;
  }
  if ( v5 < (unsigned __int64)v29 )
  {
LABEL_3:
    v16 = v5;
    goto LABEL_4;
  }
  v16 = (unsigned int)v29;
LABEL_4:
  sub_AB0910((__int64)&v29, v6);
  v14 = v30;
  if ( v30 > 0x40 )
  {
    if ( v14 - (unsigned int)sub_C444A0(&v29) > 0x40 || v5 < *(_QWORD *)v29 )
    {
      if ( !v29 )
        goto LABEL_7;
    }
    else
    {
      LODWORD(v5) = *(_QWORD *)v29;
    }
    j_j___libc_free_0_0(v29);
  }
  else
  {
    v8 = v5 < (unsigned __int64)v29;
    LODWORD(v5) = (_DWORD)v29;
    if ( v8 )
      LODWORD(v5) = v7;
  }
LABEL_7:
  sub_AB14C0((__int64)&v17, a2);
  sub_AB13A0((__int64)&v19, a2);
  v9 = 1LL << ((unsigned __int8)v18 - 1);
  if ( v18 > 0x40 )
  {
    if ( (*(_QWORD *)(v17 + 8LL * ((v18 - 1) >> 6)) & v9) != 0 )
      goto LABEL_9;
LABEL_27:
    sub_AADC60(a1, (__int64)&v17, (__int64)&v19, v16, v5);
    goto LABEL_13;
  }
  if ( (v17 & v9) == 0 )
    goto LABEL_27;
LABEL_9:
  v10 = v19;
  if ( v20 > 0x40 )
    v10 = *(_QWORD *)(v19 + 8LL * ((v20 - 1) >> 6));
  if ( (v10 & (1LL << ((unsigned __int8)v20 - 1))) != 0 )
  {
    sub_AAE090(a1, (__int64)&v17, (__int64)&v19, v16, v5);
  }
  else
  {
    sub_9691E0((__int64)&v21, v7, 0, 0, 0);
    sub_AADC60((__int64)&v25, (__int64)&v21, (__int64)&v19, v16, v5);
    sub_9691E0((__int64)&v23, v7, -1, 1u, 0);
    sub_AAE090((__int64)&v29, (__int64)&v17, (__int64)&v23, v16, v5);
    sub_AB3510(a1, (__int64)&v25, (__int64)&v29, 2u);
    if ( v32 > 0x40 && v31 )
      j_j___libc_free_0_0(v31);
    if ( v30 > 0x40 && v29 )
      j_j___libc_free_0_0(v29);
    if ( v24 > 0x40 && v23 )
      j_j___libc_free_0_0(v23);
    if ( v28 > 0x40 && v27 )
      j_j___libc_free_0_0(v27);
    if ( v26 > 0x40 && v25 )
      j_j___libc_free_0_0(v25);
    if ( v22 > 0x40 && v21 )
      j_j___libc_free_0_0(v21);
  }
LABEL_13:
  if ( v20 > 0x40 && v19 )
    j_j___libc_free_0_0(v19);
  if ( v18 > 0x40 && v17 )
    j_j___libc_free_0_0(v17);
  return a1;
}
