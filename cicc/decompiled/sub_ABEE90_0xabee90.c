// Function: sub_ABEE90
// Address: 0xabee90
//
__int64 __fastcall sub_ABEE90(__int64 *a1, unsigned __int64 a2)
{
  unsigned int v2; // r12d
  __int64 *v4; // r8
  int v5; // eax
  __int64 *v6; // r8
  unsigned int v7; // r12d
  unsigned __int64 v8; // rsi
  __int64 v9; // r14
  __int64 *v10; // rax
  unsigned int v11; // r14d
  __int64 *v12; // rsi
  int v13; // eax
  __int64 *v14; // [rsp+10h] [rbp-B0h]
  __int64 *v15; // [rsp+18h] [rbp-A8h]
  __int64 *v16; // [rsp+18h] [rbp-A8h]
  __int64 *v18; // [rsp+28h] [rbp-98h]
  __int64 v19; // [rsp+30h] [rbp-90h] BYREF
  unsigned int v20; // [rsp+38h] [rbp-88h]
  __int64 v21; // [rsp+40h] [rbp-80h] BYREF
  unsigned int v22; // [rsp+48h] [rbp-78h]
  __int64 v23; // [rsp+50h] [rbp-70h] BYREF
  unsigned int v24; // [rsp+58h] [rbp-68h]
  __int64 v25; // [rsp+60h] [rbp-60h] BYREF
  unsigned int v26; // [rsp+68h] [rbp-58h]
  __int64 v27; // [rsp+70h] [rbp-50h] BYREF
  unsigned int v28; // [rsp+78h] [rbp-48h]
  __int64 v29; // [rsp+80h] [rbp-40h] BYREF
  unsigned int v30; // [rsp+88h] [rbp-38h]

  v2 = 1;
  if ( !a2 )
    return v2;
  v4 = a1;
  v20 = *((_DWORD *)a1 + 2);
  if ( v20 > 0x40 )
  {
    sub_C43780(&v19, a1);
    v4 = a1;
  }
  else
  {
    v19 = *a1;
  }
  v15 = v4;
  v18 = a1 + 2;
  v22 = *((_DWORD *)a1 + 6);
  if ( v22 > 0x40 )
  {
    sub_C43780(&v21, v18);
    v2 = 0;
    v13 = sub_C4C880(&v19, &v21);
    v6 = v15;
    if ( v13 >= 0 )
      goto LABEL_47;
  }
  else
  {
    v2 = 0;
    v21 = a1[2];
    v5 = sub_C4C880(&v19, &v21);
    v6 = v15;
    if ( v5 >= 0 )
      goto LABEL_50;
  }
  if ( a2 == 1 )
  {
LABEL_58:
    v2 = 1;
    goto LABEL_47;
  }
  v7 = 0;
  v8 = 1;
  v9 = 0;
  while ( 1 )
  {
    v12 = &a1[4 * v8];
    v24 = *((_DWORD *)v12 + 2);
    if ( v24 <= 0x40 )
    {
      v23 = *v12;
      v26 = *((_DWORD *)v12 + 6);
      if ( v26 <= 0x40 )
        goto LABEL_10;
    }
    else
    {
      v14 = v6;
      sub_C43780(&v23, v12);
      v6 = v14;
      v26 = *((_DWORD *)v12 + 6);
      if ( v26 <= 0x40 )
      {
LABEL_10:
        v25 = v12[2];
        goto LABEL_11;
      }
    }
    v16 = v6;
    sub_C43780(&v25, v12 + 2);
    v6 = v16;
LABEL_11:
    v10 = &a1[4 * v9];
    v28 = *((_DWORD *)v10 + 2);
    if ( v28 > 0x40 )
    {
      sub_C43780(&v27, v6);
      v10 = &a1[4 * v9];
    }
    else
    {
      v27 = *v10;
    }
    v11 = *((_DWORD *)v10 + 6);
    v30 = v11;
    if ( v11 > 0x40 )
    {
      sub_C43780(&v29, v18);
      v11 = v30;
    }
    else
    {
      v29 = v10[2];
    }
    if ( (int)sub_C4C880(&v23, &v25) >= 0 || (int)sub_C4C880(&v23, &v29) <= 0 )
      break;
    if ( v11 > 0x40 && v29 )
      j_j___libc_free_0_0(v29);
    if ( v28 > 0x40 && v27 )
      j_j___libc_free_0_0(v27);
    if ( v26 > 0x40 && v25 )
      j_j___libc_free_0_0(v25);
    if ( v24 > 0x40 && v23 )
      j_j___libc_free_0_0(v23);
    v8 = v7 + 2;
    ++v7;
    if ( v8 >= a2 )
      goto LABEL_58;
    v9 = v7;
    v6 = &a1[4 * v7];
    v18 = v6 + 2;
  }
  if ( v11 > 0x40 && v29 )
    j_j___libc_free_0_0(v29);
  if ( v28 > 0x40 && v27 )
    j_j___libc_free_0_0(v27);
  if ( v26 > 0x40 && v25 )
    j_j___libc_free_0_0(v25);
  if ( v24 > 0x40 && v23 )
    j_j___libc_free_0_0(v23);
  v2 = 0;
LABEL_47:
  if ( v22 > 0x40 && v21 )
    j_j___libc_free_0_0(v21);
LABEL_50:
  if ( v20 > 0x40 && v19 )
    j_j___libc_free_0_0(v19);
  return v2;
}
