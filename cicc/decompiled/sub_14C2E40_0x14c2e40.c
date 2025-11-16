// Function: sub_14C2E40
// Address: 0x14c2e40
//
__int64 __fastcall sub_14C2E40(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v10; // eax
  __int64 *v11; // r10
  unsigned int v12; // eax
  unsigned int v13; // r8d
  __int64 result; // rax
  __int64 v15; // rdx
  __int64 v16; // r8
  __int64 v17; // rdx
  __int64 v18; // rdx
  unsigned int v19; // [rsp+8h] [rbp-78h]
  unsigned int v20; // [rsp+8h] [rbp-78h]
  unsigned int v21; // [rsp+8h] [rbp-78h]
  unsigned int v22; // [rsp+8h] [rbp-78h]
  __int64 v23; // [rsp+10h] [rbp-70h] BYREF
  unsigned int v24; // [rsp+18h] [rbp-68h]
  __int64 v25; // [rsp+20h] [rbp-60h]
  unsigned int v26; // [rsp+28h] [rbp-58h]
  __int64 v27; // [rsp+30h] [rbp-50h] BYREF
  unsigned int v28; // [rsp+38h] [rbp-48h]
  __int64 v29; // [rsp+40h] [rbp-40h]
  unsigned int v30; // [rsp+48h] [rbp-38h]

  v10 = sub_14C23D0(a1, a3, 0, a4, a5, a6);
  v11 = (__int64 *)a1;
  if ( v10 > 1 )
  {
    v12 = sub_14C23D0((__int64)a2, a3, 0, a4, a5, a6);
    v11 = (__int64 *)a1;
    v13 = v12;
    result = 2;
    if ( v13 > 1 )
      return result;
  }
  sub_14C2530((__int64)&v23, v11, a3, 0, a4, a5, a6, 0);
  sub_14C2530((__int64)&v27, a2, a3, 0, a4, a5, a6, 0);
  if ( v26 <= 0x40 )
    v15 = v25;
  else
    v15 = *(_QWORD *)(v25 + 8LL * ((v26 - 1) >> 6));
  if ( (v15 & (1LL << ((unsigned __int8)v26 - 1))) == 0 )
    goto LABEL_8;
  v16 = v29;
  if ( v30 > 0x40 )
  {
    if ( (*(_QWORD *)(v29 + 8LL * ((v30 - 1) >> 6)) & (1LL << ((unsigned __int8)v30 - 1))) != 0 )
    {
      result = 2;
LABEL_16:
      if ( v16 )
      {
        v19 = result;
        j_j___libc_free_0_0(v16);
        result = v19;
      }
      goto LABEL_18;
    }
LABEL_8:
    v17 = v23;
    if ( v24 > 0x40 )
      v17 = *(_QWORD *)(v23 + 8LL * ((v24 - 1) >> 6));
    result = 1;
    if ( (v17 & (1LL << ((unsigned __int8)v24 - 1))) != 0 )
    {
      v18 = v27;
      if ( v28 > 0x40 )
        v18 = *(_QWORD *)(v27 + 8LL * ((v28 - 1) >> 6));
      result = (unsigned int)((v18 & (1LL << ((unsigned __int8)v28 - 1))) != 0) + 1;
    }
    if ( v30 <= 0x40 )
      goto LABEL_18;
    v16 = v29;
    goto LABEL_16;
  }
  result = 2;
  if ( ((1LL << ((unsigned __int8)v30 - 1)) & v29) == 0 )
    goto LABEL_8;
LABEL_18:
  if ( v28 > 0x40 && v27 )
  {
    v20 = result;
    j_j___libc_free_0_0(v27);
    result = v20;
  }
  if ( v26 > 0x40 && v25 )
  {
    v21 = result;
    j_j___libc_free_0_0(v25);
    result = v21;
  }
  if ( v24 > 0x40 )
  {
    if ( v23 )
    {
      v22 = result;
      j_j___libc_free_0_0(v23);
      return v22;
    }
  }
  return result;
}
