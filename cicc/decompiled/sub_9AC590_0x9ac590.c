// Function: sub_9AC590
// Address: 0x9ac590
//
__int64 __fastcall sub_9AC590(__int64 a1, __int64 a2, __m128i *a3, char a4)
{
  __int64 v6; // rdx
  unsigned int v7; // r12d
  __int64 v9; // rax
  __int64 v10; // [rsp+0h] [rbp-B0h] BYREF
  unsigned int v11; // [rsp+8h] [rbp-A8h]
  __int64 v12; // [rsp+10h] [rbp-A0h]
  unsigned int v13; // [rsp+18h] [rbp-98h]
  __int64 v14; // [rsp+20h] [rbp-90h] BYREF
  unsigned int v15; // [rsp+28h] [rbp-88h]
  __int64 v16; // [rsp+30h] [rbp-80h]
  unsigned int v17; // [rsp+38h] [rbp-78h]
  __int64 v18; // [rsp+40h] [rbp-70h] BYREF
  unsigned int v19; // [rsp+48h] [rbp-68h]
  __int64 v20; // [rsp+50h] [rbp-60h]
  unsigned int v21; // [rsp+58h] [rbp-58h]
  __int64 v22; // [rsp+60h] [rbp-50h] BYREF
  unsigned int v23; // [rsp+68h] [rbp-48h]
  __int64 v24; // [rsp+70h] [rbp-40h]
  unsigned int v25; // [rsp+78h] [rbp-38h]

  sub_9AC330((__int64)&v10, a1, 0, a3);
  sub_9AC330((__int64)&v14, a2, 0, a3);
  if ( !a4 )
    goto LABEL_4;
  v6 = 1LL << ((unsigned __int8)v11 - 1);
  if ( v11 > 0x40 )
  {
    if ( (*(_QWORD *)(v10 + 8LL * ((v11 - 1) >> 6)) & v6) == 0 )
      goto LABEL_4;
LABEL_30:
    v9 = 1LL << ((unsigned __int8)v15 - 1);
    if ( v15 > 0x40 )
    {
      if ( (*(_QWORD *)(v14 + 8LL * ((v15 - 1) >> 6)) & v9) == 0 )
        goto LABEL_4;
    }
    else if ( (v14 & v9) == 0 )
    {
      goto LABEL_4;
    }
    v7 = 3;
    goto LABEL_17;
  }
  if ( (v10 & v6) != 0 )
    goto LABEL_30;
LABEL_4:
  sub_AAF050(&v18, &v10, 0);
  sub_AAF050(&v22, &v14, 0);
  v7 = sub_ABE740(&v18, &v22);
  if ( v7 > 3 )
    BUG();
  if ( v25 > 0x40 && v24 )
    j_j___libc_free_0_0(v24);
  if ( v23 > 0x40 && v22 )
    j_j___libc_free_0_0(v22);
  if ( v21 > 0x40 && v20 )
    j_j___libc_free_0_0(v20);
  if ( v19 > 0x40 && v18 )
    j_j___libc_free_0_0(v18);
LABEL_17:
  if ( v17 > 0x40 && v16 )
    j_j___libc_free_0_0(v16);
  if ( v15 > 0x40 && v14 )
    j_j___libc_free_0_0(v14);
  if ( v13 > 0x40 && v12 )
    j_j___libc_free_0_0(v12);
  if ( v11 > 0x40 && v10 )
    j_j___libc_free_0_0(v10);
  return v7;
}
