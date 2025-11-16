// Function: sub_9AFD60
// Address: 0x9afd60
//
__int64 __fastcall sub_9AFD60(__int64 *a1, __int64 *a2, __int64 a3, __m128i *a4)
{
  unsigned int v5; // eax
  unsigned int v6; // r14d
  char v7; // r15
  char v8; // bl
  unsigned int v9; // eax
  __int64 v10; // rcx
  __int64 v11; // rdi
  __int64 v12; // rdx
  __int64 v15; // [rsp+10h] [rbp-90h] BYREF
  unsigned int v16; // [rsp+18h] [rbp-88h]
  __int64 v17; // [rsp+20h] [rbp-80h]
  unsigned int v18; // [rsp+28h] [rbp-78h]
  __int64 v19; // [rsp+30h] [rbp-70h] BYREF
  unsigned int v20; // [rsp+38h] [rbp-68h]
  __int64 v21; // [rsp+40h] [rbp-60h]
  unsigned int v22; // [rsp+48h] [rbp-58h]
  __int64 v23; // [rsp+50h] [rbp-50h] BYREF
  unsigned int v24; // [rsp+58h] [rbp-48h]
  __int64 v25; // [rsp+60h] [rbp-40h]
  unsigned int v26; // [rsp+68h] [rbp-38h]

  if ( a3 && (*(_BYTE *)(a3 + 1) & 4) != 0
    || (unsigned int)sub_9AF7E0(*a1 & 0xFFFFFFFFFFFFFFF8LL, 0, a4) > 1
    && (unsigned int)sub_9AF7E0(*a2 & 0xFFFFFFFFFFFFFFF8LL, 0, a4) > 1 )
  {
    return 3;
  }
  sub_9AC780((__int64)&v15, a1, 1u, a4);
  sub_9AC780((__int64)&v19, a2, 1u, a4);
  v5 = sub_ABDC10(&v15, &v19);
  v6 = v5;
  if ( v5 > 3 )
    BUG();
  if ( v5 != 2 || !a3 )
  {
LABEL_22:
    if ( v22 > 0x40 )
      goto LABEL_23;
    goto LABEL_25;
  }
  v7 = sub_AB0760(&v15);
  if ( !v7 )
    v7 = sub_AB0760(&v19);
  if ( (unsigned __int8)sub_AB06D0(&v15) || (v8 = sub_AB06D0(&v19)) != 0 )
  {
    v8 = 1;
    goto LABEL_12;
  }
  if ( v7 )
  {
LABEL_12:
    sub_9878D0((__int64)&v23, v16);
    sub_99B5E0(a3, (__int64)&v23, 0, a4->m128i_i64, (__int64)&v23);
    v9 = v24;
    if ( v24 <= 0x40 )
      v10 = v23;
    else
      v10 = *(_QWORD *)(v23 + 8LL * ((v24 - 1) >> 6));
    if ( (v10 & (1LL << ((unsigned __int8)v24 - 1))) != 0 && v7 )
    {
      if ( v26 <= 0x40 )
      {
LABEL_49:
        if ( v9 > 0x40 && v23 )
          j_j___libc_free_0_0(v23);
        v6 = 3;
        goto LABEL_22;
      }
      v11 = v25;
    }
    else
    {
      v11 = v25;
      v12 = 1LL << ((unsigned __int8)v26 - 1);
      if ( v26 <= 0x40 )
      {
        if ( (v12 & v25) == 0 || !v8 )
        {
LABEL_19:
          if ( v9 > 0x40 && v23 )
            j_j___libc_free_0_0(v23);
          goto LABEL_22;
        }
        goto LABEL_49;
      }
      if ( (*(_QWORD *)(v25 + 8LL * ((v26 - 1) >> 6)) & v12) == 0 || !v8 )
      {
        if ( v25 )
        {
          j_j___libc_free_0_0(v25);
          v9 = v24;
        }
        goto LABEL_19;
      }
    }
    if ( v11 )
    {
      j_j___libc_free_0_0(v11);
      v9 = v24;
    }
    goto LABEL_49;
  }
  if ( v22 > 0x40 )
  {
LABEL_23:
    if ( v21 )
      j_j___libc_free_0_0(v21);
  }
LABEL_25:
  if ( v20 > 0x40 && v19 )
    j_j___libc_free_0_0(v19);
  if ( v18 > 0x40 && v17 )
    j_j___libc_free_0_0(v17);
  if ( v16 > 0x40 && v15 )
    j_j___libc_free_0_0(v15);
  return v6;
}
