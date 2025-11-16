// Function: sub_14C28B0
// Address: 0x14c28b0
//
__int64 __fastcall sub_14C28B0(__int64 *a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v10; // eax
  __int64 v11; // rcx
  __int64 v12; // rdi
  __int64 v13; // rdx
  __int64 v14; // rax
  unsigned int v15; // esi
  __int64 v16; // rdi
  __int64 v17; // rax
  unsigned int v18; // r12d
  unsigned int v19; // eax
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v23; // [rsp+0h] [rbp-70h] BYREF
  unsigned int v24; // [rsp+8h] [rbp-68h]
  __int64 v25; // [rsp+10h] [rbp-60h]
  unsigned int v26; // [rsp+18h] [rbp-58h]
  __int64 v27; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v28; // [rsp+28h] [rbp-48h]
  __int64 v29; // [rsp+30h] [rbp-40h]
  unsigned int v30; // [rsp+38h] [rbp-38h]

  sub_14C2530((__int64)&v23, a1, a3, 0, a4, a5, a6, 0);
  v10 = v24;
  if ( v24 > 0x40 )
    v11 = *(_QWORD *)(v23 + 8LL * ((v24 - 1) >> 6));
  else
    v11 = v23;
  if ( (v11 & (1LL << ((unsigned __int8)v24 - 1))) == 0 )
  {
    v12 = v25;
    v13 = 1LL << ((unsigned __int8)v26 - 1);
    if ( v26 > 0x40 )
    {
      if ( (*(_QWORD *)(v25 + 8LL * ((v26 - 1) >> 6)) & v13) == 0 )
      {
        v18 = 1;
        goto LABEL_34;
      }
    }
    else if ( (v13 & v25) == 0 )
    {
      v18 = 1;
      goto LABEL_37;
    }
  }
  sub_14C2530((__int64)&v27, a2, a3, 0, a4, a5, a6, 0);
  v14 = 1LL << ((unsigned __int8)v26 - 1);
  if ( v26 <= 0x40 )
  {
    v15 = v30;
    if ( (v25 & v14) == 0 )
      goto LABEL_20;
  }
  else
  {
    v15 = v30;
    if ( (*(_QWORD *)(v25 + 8LL * ((v26 - 1) >> 6)) & v14) == 0 )
      goto LABEL_20;
  }
  v16 = v29;
  v17 = 1LL << ((unsigned __int8)v15 - 1);
  if ( v15 <= 0x40 )
  {
    if ( (v17 & v29) != 0 )
    {
      v19 = v28;
      v18 = 0;
      goto LABEL_14;
    }
  }
  else if ( (*(_QWORD *)(v29 + 8LL * ((v15 - 1) >> 6)) & v17) != 0 )
  {
    v18 = 0;
LABEL_11:
    if ( v16 )
      j_j___libc_free_0_0(v16);
    v19 = v28;
    goto LABEL_14;
  }
LABEL_20:
  v20 = v23;
  if ( v24 > 0x40 )
    v20 = *(_QWORD *)(v23 + 8LL * ((v24 - 1) >> 6));
  if ( (v20 & (1LL << ((unsigned __int8)v24 - 1))) == 0 )
    goto LABEL_25;
  v19 = v28;
  v21 = 1LL << ((unsigned __int8)v28 - 1);
  if ( v28 <= 0x40 )
  {
    if ( (v27 & v21) == 0 )
      goto LABEL_25;
    goto LABEL_43;
  }
  if ( (*(_QWORD *)(v27 + 8LL * ((v28 - 1) >> 6)) & v21) != 0 )
  {
LABEL_43:
    v18 = 2;
    if ( v15 > 0x40 )
    {
      v16 = v29;
      goto LABEL_11;
    }
LABEL_14:
    if ( v19 > 0x40 && v27 )
      j_j___libc_free_0_0(v27);
    goto LABEL_32;
  }
LABEL_25:
  if ( v15 > 0x40 && v29 )
    j_j___libc_free_0_0(v29);
  if ( v28 > 0x40 && v27 )
    j_j___libc_free_0_0(v27);
  v18 = 1;
LABEL_32:
  if ( v26 > 0x40 )
  {
    v12 = v25;
LABEL_34:
    if ( v12 )
      j_j___libc_free_0_0(v12);
  }
  v10 = v24;
LABEL_37:
  if ( v10 > 0x40 && v23 )
    j_j___libc_free_0_0(v23);
  return v18;
}
